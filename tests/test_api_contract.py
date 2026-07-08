"""SSE 이벤트 계약 테스트 — 파이프라인/LLM 목, 모델 로딩 없음.
api/sse.py ↔ static/js/app.js 가 미러하는 계약을 고정한다."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from panrye.api.main import app
from panrye.core.types import StageEvent

# lifespan(init_db 등) 없이 라우트만 검증
client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_sse_appstatus():
    # sse-starlette의 AppStatus.should_exit_event는 최초 이벤트루프에 바인딩됨.
    # TestClient가 테스트마다 새 루프를 만들므로 매번 리셋해야 한다.
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit_event = None


def fake_stream_retrieval(user_query: str):
    for stage in ["classify", "reformulate", "retrieve", "summarize"]:
        yield StageEvent(stage=stage, status="start")
        payload = {"domain": "민사", "confidence": 0.8} if stage == "classify" else {}
        yield StageEvent(stage=stage, status="done", elapsed_ms=12.3, payload=payload)
    return {
        "user_query": user_query,
        "domain": "민사",
        "domain_confidence": 0.8,
        "rewritten_query": "금전채무 불이행 민법",
        "retrieved_chunks": [object()],
        "summaries": [{
            "case_id": "1", "case_number": "2023다1111", "case_name": "손해배상(기)",
            "court": "대법원", "date": "2023.05.15", "domain": "민사",
            "summary": "요약문", "key_statutes": "민법 제390조",
            "verdict_snippet": "지급하라", "full_text_snippet": "본문 발췌",
            "rerank_score": 0.9, "source": "law.go.kr",
        }],
        "context": "판례 컨텍스트",
    }


def fake_stream_answer(user_query: str, context: str):
    yield from ["답변 ", "토큰"]


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    current_event = None
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and current_event:
            events.append((current_event, json.loads(line.split(":", 1)[1].strip())))
            current_event = None
    return events


@pytest.fixture()
def sse_events():
    with (
        patch("panrye.graph.pipeline.stream_retrieval", fake_stream_retrieval),
        patch("panrye.agents.generator.stream_answer", fake_stream_answer),
        patch("panrye.api.main.log_query", return_value=42),
    ):
        with client.stream("GET", "/api/stream", params={"query": "친구가 돈을 안 갚아요"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
    return _parse_sse(body)


def test_event_order(sse_events):
    names = [name for name, _ in sse_events]
    # stage들 사이에 domain이 오고, context → answer_chunk* → done 순서
    assert names[0] == "stage"
    assert "domain" in names
    assert names.index("domain") < names.index("context")
    chunk_indices = [i for i, n in enumerate(names) if n == "answer_chunk"]
    assert chunk_indices, "answer_chunk 이벤트 필요"
    assert names.index("context") < chunk_indices[0]
    assert names[-1] == "done"


def test_stage_events_cover_all_stages(sse_events):
    stages_done = {d["stage"] for n, d in sse_events if n == "stage" and d["status"] == "done"}
    assert stages_done == {"classify", "reformulate", "retrieve", "summarize", "generate"}
    for _, d in [e for e in sse_events if e[0] == "stage"]:
        assert set(d) == {"stage", "status", "ms", "detail"}


def test_context_card_schema(sse_events):
    context = next(d for n, d in sse_events if n == "context")
    assert context["retrieved_count"] == 1
    card = context["cards"][0]
    assert set(card) == {
        "case_name", "case_number", "court", "date", "domain", "statutes",
        "verdict", "summary", "full_text_snippet", "score", "source",
    }
    assert card["case_number"] == "2023다1111"


def test_done_carries_query_id(sse_events):
    done = next(d for n, d in sse_events if n == "done")
    assert done["query_id"] == 42
    assert "latency_ms" in done


def test_query_length_validation():
    r = client.get("/api/stream", params={"query": "짧다"})
    assert r.status_code == 400


def test_error_event_on_pipeline_failure():
    def boom(user_query: str):
        raise RuntimeError("인덱스 없음")
        yield  # generator로 만들기 (실행 안 됨)

    with (
        patch("panrye.graph.pipeline.stream_retrieval", boom),
        patch("panrye.agents.generator.stream_answer", fake_stream_answer),
    ):
        with client.stream("GET", "/api/stream", params={"query": "친구가 돈을 안 갚아요"}) as r:
            body = "".join(r.iter_text())
    events = _parse_sse(body)
    assert events[-1][0] == "error"
    assert "stage" in events[-1][1] and "message" in events[-1][1]


def test_feedback_endpoint():
    with patch("panrye.api.main.update_feedback") as mock_fb:
        r = client.post("/api/feedback", json={"query_id": 7, "helpful": True})
    assert r.status_code == 200
    mock_fb.assert_called_once_with(7, True)


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"
