"""파이프라인 전역에서 공유하는 타입 정의."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class RetrievedChunk:
    chunk_id: str
    case_id: str
    case_name: str
    court: str
    date: str
    domain: str
    text: str
    statutes: list[str]
    source: str
    verdict: str
    rrf_score: float
    rerank_score: float = 0.0


class PipelineState(TypedDict):
    # Input
    user_query: str
    # classify
    domain: str
    domain_confidence: float
    # reformulate
    rewritten_query: str
    hyde_document: str
    # retrieve
    retrieved_chunks: list[RetrievedChunk]
    # summarize
    summaries: list[dict]
    context: str
    # generate
    final_answer: str
    # meta
    stage_timings: dict[str, float]
    pipeline_error: str | None
    log_id: int | None


@dataclass
class StageEvent:
    """SSE로 흘려보내는 파이프라인 단계 이벤트."""

    stage: str  # classify | reformulate | retrieve | summarize | generate
    status: str  # start | done
    elapsed_ms: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
