"""
/api/stream SSE 이벤트 계약 — 단일 소스.
이벤트 순서: stage(start/done)* → domain → context → stage(generate) → answer_chunk* → done
오류 시 어느 시점이든: error {stage, message}

static/js/app.js 의 EVENT 상수와 tests/test_api_contract.py 가 이 계약을 미러링한다.
"""
from __future__ import annotations

import json

EVENT_STAGE = "stage"
EVENT_DOMAIN = "domain"
EVENT_CONTEXT = "context"
EVENT_ANSWER_CHUNK = "answer_chunk"
EVENT_DONE = "done"
EVENT_ERROR = "error"

STAGES = ["classify", "reformulate", "retrieve", "summarize", "generate"]


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def stage_event(stage: str, status: str, ms: float = 0.0, detail: dict | None = None) -> dict:
    return _sse(EVENT_STAGE, {
        "stage": stage, "status": status, "ms": ms, "detail": detail or {},
    })


def domain_event(domain: str, confidence: float) -> dict:
    return _sse(EVENT_DOMAIN, {"domain": domain, "confidence": confidence})


def context_event(summaries: list[dict], retrieved_count: int) -> dict:
    return _sse(EVENT_CONTEXT, {
        "cards": [_card(s) for s in summaries],
        "retrieved_count": retrieved_count,
    })


def _card(s: dict) -> dict:
    """증거 판례 카드 payload."""
    return {
        "case_name": s.get("case_name", ""),
        "case_number": s.get("case_number", ""),
        "court": s.get("court", ""),
        "date": s.get("date", ""),
        "domain": s.get("domain", ""),
        "statutes": s.get("key_statutes", ""),
        "verdict": s.get("verdict_snippet", ""),
        "summary": s.get("summary", ""),
        "full_text_snippet": s.get("full_text_snippet", ""),
        "score": s.get("rerank_score", 0.0),
        "source": s.get("source", ""),
    }


def answer_chunk_event(text: str) -> dict:
    return _sse(EVENT_ANSWER_CHUNK, {"text": text})


def done_event(query_id: int | None, latency_ms: float) -> dict:
    return _sse(EVENT_DONE, {"query_id": query_id, "latency_ms": round(latency_ms, 1)})


def error_event(stage: str, message: str) -> dict:
    return _sse(EVENT_ERROR, {"stage": stage, "message": message})
