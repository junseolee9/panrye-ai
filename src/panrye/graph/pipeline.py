"""
LangGraph Multi-Agent Pipeline.
State machine: Classify → Reformulate → Retrieve → Summarize → (Generate)

- run_pipeline(): 동기 전체 실행 (/api/query, eval)
- stream_retrieval(): generate 제외 그래프를 stream_mode="updates"로 돌리며
  노드별 StageEvent를 yield — SSE 스테이지 스트리밍용.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from functools import lru_cache

from panrye.agents.classifier import classify_domain
from panrye.agents.generator import generate_answer
from panrye.agents.reformulator import reformulate
from panrye.agents.retriever import retrieve
from panrye.agents.summarizer import format_context_for_llm, summarize_precedents
from panrye.core.types import PipelineState, StageEvent
from panrye.storage.db import log_query

logger = logging.getLogger(__name__)

RETRIEVAL_STAGES = ["classify", "reformulate", "retrieve", "summarize"]


def _timed(stage: str, updates: dict, start: float) -> dict:
    timings = {stage: (time.time() - start) * 1000}
    return {**updates, "stage_timings": timings}


def node_classify(state: PipelineState) -> dict:
    start = time.time()
    result = classify_domain(state["user_query"])
    return _timed("classify", {
        "domain": result["domain"],
        "domain_confidence": result["confidence"],
    }, start)


def node_reformulate(state: PipelineState) -> dict:
    start = time.time()
    result = reformulate(state["user_query"], state["domain"])
    return _timed("reformulate", {
        "rewritten_query": result["rewritten_query"],
        "hyde_document": result["hyde_document"],
    }, start)


def node_retrieve(state: PipelineState) -> dict:
    start = time.time()
    chunks = retrieve(
        query=state["rewritten_query"],
        hyde_document=state["hyde_document"],
        domain=state["domain"],
    )
    return _timed("retrieve", {"retrieved_chunks": chunks}, start)


def node_summarize(state: PipelineState) -> dict:
    start = time.time()
    summaries = summarize_precedents(state["retrieved_chunks"])
    context = format_context_for_llm(summaries)
    return _timed("summarize", {"summaries": summaries, "context": context}, start)


def node_generate(state: PipelineState) -> dict:
    start = time.time()
    answer = generate_answer(state["user_query"], state["context"])
    return _timed("generate", {"final_answer": answer}, start)


@lru_cache(maxsize=2)
def build_graph(include_generate: bool = True):
    from langgraph.graph import END, StateGraph

    graph = StateGraph(PipelineState)
    graph.add_node("classify", node_classify)
    graph.add_node("reformulate", node_reformulate)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("summarize", node_summarize)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "reformulate")
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("retrieve", "summarize")

    if include_generate:
        graph.add_node("generate", node_generate)
        graph.add_edge("summarize", "generate")
        graph.add_edge("generate", END)
    else:
        graph.add_edge("summarize", END)

    return graph.compile()


def _initial_state(user_query: str) -> PipelineState:
    return {
        "user_query": user_query,
        "domain": "",
        "domain_confidence": 0.0,
        "rewritten_query": "",
        "hyde_document": "",
        "retrieved_chunks": [],
        "summaries": [],
        "context": "",
        "final_answer": "",
        "stage_timings": {},
        "pipeline_error": None,
        "log_id": None,
    }


def run_pipeline(user_query: str) -> dict:
    """동기 전체 실행. 전체 state 반환 + SQLite 로깅."""
    pipeline = build_graph(include_generate=True)
    initial_state = _initial_state(user_query)
    start = time.time()
    try:
        result = pipeline.invoke(initial_state)
    except Exception as e:
        logger.exception("파이프라인 실행 실패")
        return {
            **initial_state,
            "final_answer": "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "pipeline_error": str(e),
        }

    latency_ms = (time.time() - start) * 1000
    try:
        result["log_id"] = log_query(
            user_query=user_query,
            domain=result["domain"],
            rewritten_query=result["rewritten_query"],
            retrieved_count=len(result["retrieved_chunks"]),
            summaries=result["summaries"],
            final_answer=result["final_answer"],
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("쿼리 로깅 실패")
    return result


def stream_retrieval(user_query: str) -> Generator[StageEvent, None, dict]:
    """
    검색 파이프라인(생성 제외)을 스테이지 이벤트로 스트리밍.
    yield: StageEvent(start/done, 노드별 레이턴시, 노드 출력 payload)
    return: 최종 state (generate 입력용) — yield from 으로 수신.
    """
    pipeline = build_graph(include_generate=False)
    state: dict = _initial_state(user_query)

    next_idx = 0
    yield StageEvent(stage=RETRIEVAL_STAGES[0], status="start")

    for update in pipeline.stream(state, stream_mode="updates"):
        for node_name, node_output in update.items():
            if node_name not in RETRIEVAL_STAGES:
                continue
            timings = node_output.get("stage_timings", {})
            state.update(node_output)
            yield StageEvent(
                stage=node_name,
                status="done",
                elapsed_ms=round(timings.get(node_name, 0.0), 1),
                payload=_stage_payload(node_name, state),
            )
            next_idx = RETRIEVAL_STAGES.index(node_name) + 1
            if next_idx < len(RETRIEVAL_STAGES):
                yield StageEvent(stage=RETRIEVAL_STAGES[next_idx], status="start")

    return state


def _stage_payload(stage: str, state: dict) -> dict:
    """스테이지별 UI에 흘려보낼 요약 payload."""
    if stage == "classify":
        return {"domain": state["domain"], "confidence": state["domain_confidence"]}
    if stage == "reformulate":
        return {"rewritten_query": state["rewritten_query"]}
    if stage == "retrieve":
        return {"retrieved_count": len(state["retrieved_chunks"])}
    if stage == "summarize":
        return {"case_count": len(state["summaries"])}
    return {}
