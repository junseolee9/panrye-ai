"""Reciprocal Rank Fusion 순수 함수 검증."""
import pytest

from panrye.agents.retriever import reciprocal_rank_fusion


def _chunks(ids: list[str]) -> list[dict]:
    return [{"chunk_id": cid, "text": f"본문 {cid}"} for cid in ids]


def test_rrf_math_k60():
    dense = _chunks(["a", "b"])
    bm25 = _chunks(["b", "c"])
    fused = reciprocal_rank_fusion(dense, bm25, k=60)
    scores = {c["chunk_id"]: c["rrf_score"] for c in fused}

    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)  # dense 2위 + bm25 1위
    assert scores["c"] == pytest.approx(1 / 62)


def test_overlap_ranks_first():
    # 양쪽 모두에 등장한 문서가 최상위
    dense = _chunks(["x", "shared"])
    bm25 = _chunks(["shared", "y"])
    fused = reciprocal_rank_fusion(dense, bm25)
    assert fused[0]["chunk_id"] == "shared"


def test_rank_stability_desc():
    dense = _chunks(["a", "b", "c", "d"])
    fused = reciprocal_rank_fusion(dense, [])
    scores = [c["rrf_score"] for c in fused]
    assert scores == sorted(scores, reverse=True)
    assert [c["chunk_id"] for c in fused] == ["a", "b", "c", "d"]


def test_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []
