"""
FastAPI 백엔드.
- /            : SPA (static/)
- POST /api/query    : 동기 답변 (eval·프로그래밍 용도)
- GET  /api/stream   : SSE 스테이지+답변 스트리밍 (UI 메인 경로)
- POST /api/feedback : 사용자 피드백
- GET  /api/stats    : 운영 통계
- GET  /api/health   : 헬스체크
"""
from __future__ import annotations

import logging
import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from panrye.api import sse
from panrye.api.schemas import FeedbackRequest, QueryRequest, QueryResponse
from panrye.config import get_settings
from panrye.storage.db import (
    get_eval_stats,
    get_query_stats,
    get_recent_queries,
    init_db,
    log_query,
    update_feedback,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[3] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()
    _ensure_artifacts()
    yield


def _ensure_artifacts() -> None:
    """인덱스 아티팩트 확인. 없으면 HF Dataset에서 부트스트랩 (배포 환경)."""
    settings = get_settings()
    if settings.chroma_dir.exists() and settings.bm25_path.exists():
        logger.info("인덱스 아티팩트 확인됨.")
        return
    if settings.index_repo_id:
        from panrye.storage.artifacts import download_artifacts

        logger.info(f"인덱스 아티팩트 없음 → {settings.index_repo_id} 에서 다운로드")
        download_artifacts()
    else:
        logger.warning(
            "인덱스 아티팩트 없음. scripts/build_index.py 실행 또는 INDEX_REPO_ID 설정 필요."
        )


app = FastAPI(
    title="판례.ai",
    description="한국 판례 기반 법률 상담 RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# SPA는 이 서버가 직접 서빙하므로 교차 출처는 기본 차단. CORS_ORIGINS로만 연다.
_cors_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# IP별 최근 요청 시각. 단일 프로세스 인메모리 — 데모 규모에 맞춘 최소 구현.
_recent_hits: dict[str, deque[float]] = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    """IP당 분당 요청 수 제한. 초과 시 429."""
    limit = get_settings().rate_limit_per_min
    if limit <= 0:
        return

    now = time.time()
    ip = request.client.host if request.client else "unknown"
    hits = _recent_hits[ip]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= limit:
        raise HTTPException(
            status_code=429, detail="요청이 너무 잦습니다. 잠시 후 다시 시도해주세요."
        )
    hits.append(now)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "panrye.ai"}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    from panrye.graph.pipeline import run_pipeline

    _rate_limit(request)

    start = time.time()
    result = run_pipeline(req.query)
    latency = (time.time() - start) * 1000

    if result.get("pipeline_error"):
        raise HTTPException(status_code=500, detail=result["pipeline_error"])

    return QueryResponse(
        query_id=result.get("log_id"),
        domain=result["domain"],
        domain_confidence=result["domain_confidence"],
        rewritten_query=result["rewritten_query"],
        retrieved_count=len(result["retrieved_chunks"]),
        summaries=result["summaries"],
        answer=result["final_answer"],
        latency_ms=latency,
    )


def _retrieval_events(user_query: str) -> Generator[tuple[str, object], None, None]:
    """stream_retrieval의 StageEvent와 최종 state를 하나의 제너레이터로 평탄화.
    iterate_in_threadpool은 return value를 못 받으므로 ('final', state)로 넘긴다."""
    from panrye.graph.pipeline import stream_retrieval

    gen = stream_retrieval(user_query)
    while True:
        try:
            yield ("event", next(gen))
        except StopIteration as stop:
            yield ("final", stop.value)
            return


@app.get("/api/stream")
async def stream(query: str, request: Request):
    """SSE: 스테이지 진행 → 도메인/근거 카드 → 답변 토큰 스트림."""
    _rate_limit(request)
    if not 5 <= len(query) <= 1000:
        raise HTTPException(status_code=400, detail="질문은 5자 이상 1000자 이하여야 합니다.")

    async def event_generator() -> AsyncGenerator[dict, None]:
        from panrye.agents.generator import stream_answer

        start = time.time()
        state: dict | None = None
        current_stage = "classify"

        try:
            # 1. 검색 파이프라인: 노드별 stage 이벤트 중계
            async for kind, item in iterate_in_threadpool(_retrieval_events(query)):
                if kind == "final":
                    state = item
                    break
                current_stage = item.stage
                yield sse.stage_event(item.stage, item.status, item.elapsed_ms, item.payload)
                if item.stage == "classify" and item.status == "done":
                    yield sse.domain_event(
                        item.payload.get("domain", ""), item.payload.get("confidence", 0.0)
                    )

            if state is None:
                raise RuntimeError("검색 파이프라인이 결과를 반환하지 않았습니다.")

            yield sse.context_event(state["summaries"], len(state["retrieved_chunks"]))

            # 2. 답변 토큰 스트리밍
            current_stage = "generate"
            yield sse.stage_event("generate", "start")
            gen_start = time.time()
            answer_parts: list[str] = []
            async for chunk in iterate_in_threadpool(stream_answer(query, state["context"])):
                answer_parts.append(chunk)
                yield sse.answer_chunk_event(chunk)
            yield sse.stage_event("generate", "done", (time.time() - gen_start) * 1000)

            # 3. 로깅
            log_id = await run_in_threadpool(
                log_query,
                user_query=query,
                domain=state["domain"],
                rewritten_query=state["rewritten_query"],
                retrieved_count=len(state["retrieved_chunks"]),
                summaries=state["summaries"],
                final_answer="".join(answer_parts),
                latency_ms=(time.time() - start) * 1000,
            )
            yield sse.done_event(log_id, (time.time() - start) * 1000)

        except Exception as e:
            logger.exception("스트리밍 파이프라인 실패")
            yield sse.error_event(current_stage, str(e))

    return EventSourceResponse(event_generator())


@app.post("/api/feedback")
def feedback(req: FeedbackRequest, request: Request):
    _rate_limit(request)
    # update_feedback은 아직 피드백이 없는 행만 갱신 — 남의 평가를 뒤집을 수 없다
    update_feedback(req.query_id, req.helpful)
    return {"status": "ok"}


@app.get("/api/stats")
def stats(request: Request, token: str = ""):
    """운영 통계. 최근 질문이 섞여 있으므로 STATS_TOKEN 없이는 존재를 노출하지 않는다."""
    _rate_limit(request)
    expected = get_settings().stats_token
    if not expected or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=404, detail="Not Found")

    recent = get_recent_queries(limit=5)
    return {
        "query_stats": get_query_stats(),
        "eval_stats": get_eval_stats(),
        "recent_queries": [
            {
                "id": q["id"],
                "query": q["user_query"][:50],
                "domain": q["domain"],
                "timestamp": q["timestamp"],
            }
            for q in recent
        ],
    }


# SPA — API 라우트 뒤에 마운트 (html=True로 index.html 서빙)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
