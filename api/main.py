"""
FastAPI 백엔드.
- POST /query: 동기 답변
- GET /stream: SSE 스트리밍 답변
- GET /health: 헬스체크
- POST /feedback: 사용자 피드백
"""
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from graph.pipeline import run_pipeline, run_retrieval
from agents.generator import stream_answer
from logs.db import log_query, update_feedback, get_eval_stats, get_recent_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PanRye.ai",
    description="한국 판례 기반 법률 상담 RAG 챗봇",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000, description="사용자 법률 상황 설명")


class QueryResponse(BaseModel):
    query_id: int | None
    domain: str
    domain_confidence: float
    rewritten_query: str
    retrieved_count: int
    summaries: list[dict]
    answer: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    query_id: int
    helpful: bool


@app.get("/health")
def health():
    return {"status": "ok", "service": "PanRye.ai"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
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


@app.get("/stream")
async def stream(query: str):
    """SSE 스트리밍 답변."""
    if len(query) < 5:
        raise HTTPException(status_code=400, detail="쿼리가 너무 짧습니다.")

    async def event_generator() -> AsyncGenerator[dict, None]:
        start = time.time()

        # 1. 검색 파이프라인 (분류→재작성→검색→요약). 생성은 아래에서 스트리밍.
        yield {"event": "status", "data": json.dumps({"step": "retrieve", "message": "판례 검색 중..."}, ensure_ascii=False)}
        result = await run_in_threadpool(run_retrieval, query)

        yield {"event": "domain", "data": json.dumps({"domain": result["domain"]}, ensure_ascii=False)}
        yield {"event": "context", "data": json.dumps({
            "summaries": result["summaries"],
            "retrieved_count": len(result["retrieved_chunks"]),
        }, ensure_ascii=False)}

        # 2. 스트리밍 답변 (동기 제너레이터를 threadpool에서 반복)
        yield {"event": "status", "data": json.dumps({"step": "generate", "message": "답변 생성 중..."}, ensure_ascii=False)}
        answer_parts: list[str] = []
        async for chunk in iterate_in_threadpool(stream_answer(query, result["context"])):
            answer_parts.append(chunk)
            yield {"event": "answer_chunk", "data": json.dumps({"text": chunk}, ensure_ascii=False)}

        # 3. 로깅
        log_id = await run_in_threadpool(
            log_query,
            user_query=query,
            domain=result["domain"],
            rewritten_query=result["rewritten_query"],
            retrieved_count=len(result["retrieved_chunks"]),
            summaries=result["summaries"],
            final_answer="".join(answer_parts),
            latency_ms=(time.time() - start) * 1000,
        )
        yield {"event": "done", "data": json.dumps({"query_id": log_id})}

    return EventSourceResponse(event_generator())


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    update_feedback(req.query_id, req.helpful)
    return {"status": "ok"}


@app.get("/stats")
def stats():
    eval_stats = get_eval_stats()
    recent = get_recent_queries(limit=5)
    return {
        "eval_stats": eval_stats,
        "recent_queries": [
            {
                "id": q["id"],
                "query": q["user_query"][:50] + "...",
                "domain": q["domain"],
                "timestamp": q["timestamp"],
            }
            for q in recent
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
