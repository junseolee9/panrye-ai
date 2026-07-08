"""
LangGraph Multi-Agent Pipeline.
State machine: Classify → Reformulate → Retrieve → Summarize → Generate
"""
from __future__ import annotations
import logging
import time
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.classifier import classify_domain
from agents.reformulator import reformulate
from agents.retriever import retrieve, RetrievedChunk
from agents.summarizer import summarize_precedents, format_context_for_llm
from agents.generator import generate_answer
from logs.db import log_query

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    # Input
    user_query: str
    # Agent 1 output
    domain: str
    domain_confidence: float
    # Agent 2 output
    rewritten_query: str
    hyde_document: str
    # Agent 3 output
    retrieved_chunks: list[RetrievedChunk]
    # Agent 4 output
    summaries: list[dict]
    context: str
    # Agent 5 output
    final_answer: str
    # Meta
    pipeline_error: str | None
    log_id: int | None


def node_classify(state: PipelineState) -> dict:
    """Agent 1: 도메인 분류."""
    logger.info("[Node] Classify")
    result = classify_domain(state["user_query"])
    return {
        "domain": result["domain"],
        "domain_confidence": result["confidence"],
    }


def node_reformulate(state: PipelineState) -> dict:
    """Agent 2: 쿼리 재작성 + HyDE."""
    logger.info(f"[Node] Reformulate (domain={state['domain']})")
    result = reformulate(state["user_query"], state["domain"])
    return {
        "rewritten_query": result["rewritten_query"],
        "hyde_document": result["hyde_document"],
    }


def node_retrieve(state: PipelineState) -> dict:
    """Agent 3: Hybrid 검색."""
    logger.info("[Node] Retrieve")
    chunks = retrieve(
        query=state["rewritten_query"],
        hyde_document=state["hyde_document"],
        domain=state["domain"],
    )
    return {"retrieved_chunks": chunks}


def node_summarize(state: PipelineState) -> dict:
    """Agent 4: 판례 요약."""
    logger.info("[Node] Summarize")
    summaries = summarize_precedents(state["retrieved_chunks"])
    context = format_context_for_llm(summaries)
    return {"summaries": summaries, "context": context}


def node_generate(state: PipelineState) -> dict:
    """Agent 5: 답변 생성."""
    logger.info("[Node] Generate")
    answer = generate_answer(state["user_query"], state["context"])
    return {"final_answer": answer}


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("classify", node_classify)
    graph.add_node("reformulate", node_reformulate)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("summarize", node_summarize)
    graph.add_node("generate", node_generate)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "reformulate")
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# 싱글턴 파이프라인 인스턴스
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


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
        "pipeline_error": None,
        "log_id": None,
    }


def run_pipeline(user_query: str) -> dict:
    """동기 실행. 전체 state 반환."""
    pipeline = get_pipeline()
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


def run_retrieval(user_query: str) -> dict:
    """
    생성(Generate) 제외한 검색 파이프라인: 분류 → 재작성 → 검색 → 요약.
    스트리밍 API에서 답변 생성을 별도로 스트리밍할 때 사용.
    """
    domain_result = classify_domain(user_query)
    reform = reformulate(user_query, domain_result["domain"])
    chunks = retrieve(
        query=reform["rewritten_query"],
        hyde_document=reform["hyde_document"],
        domain=domain_result["domain"],
    )
    summaries = summarize_precedents(chunks)
    context = format_context_for_llm(summaries)
    return {
        "user_query": user_query,
        "domain": domain_result["domain"],
        "domain_confidence": domain_result["confidence"],
        "rewritten_query": reform["rewritten_query"],
        "hyde_document": reform["hyde_document"],
        "retrieved_chunks": chunks,
        "summaries": summaries,
        "context": context,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_pipeline("친구한테 300만원 빌려줬는데 1년째 안 갚아요. 어떻게 해야 하나요?")
    print("\n=== 최종 답변 ===")
    print(result["final_answer"])
    print(f"\n도메인: {result['domain']} (confidence={result['domain_confidence']:.2f})")
    print(f"재작성 쿼리: {result['rewritten_query']}")
    print(f"검색된 판례: {len(result['retrieved_chunks'])}건")
