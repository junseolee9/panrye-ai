"""API 요청/응답 Pydantic 모델."""
from __future__ import annotations

from pydantic import BaseModel, Field


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
