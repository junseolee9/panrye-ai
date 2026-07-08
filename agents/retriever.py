"""
Agent 3: Hybrid Retriever
Dense (ChromaDB) + Sparse (BM25, 형태소 토큰화) → Reciprocal Rank Fusion → Cross-encoder Reranking.
Top-20 후보 → Rerank → Top-5 최종.
"""
from __future__ import annotations
import json
import logging
import pickle
from functools import lru_cache
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from data.index_build import CHROMA_DB_PATH, BM25_INDEX_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from data.tokenizer import tokenize

logger = logging.getLogger(__name__)

TOP_K_DENSE = 20
TOP_K_BM25 = 20
TOP_K_FINAL = 5
RRF_K = 60

RERANKER_MODEL = "Dongjin-kr/ko-reranker"


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


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    logger.info(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    logger.info(f"Reranker 로딩: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL, max_length=512)



@lru_cache(maxsize=1)
def _get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


@lru_cache(maxsize=1)
def _get_bm25_index() -> dict:
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 인덱스 없음: {BM25_INDEX_PATH}")
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def _dense_search(query_text: str, domain_filter: str | None = None, top_k: int = TOP_K_DENSE) -> list[dict]:
    """ChromaDB 벡터 검색."""
    model = _get_embedding_model()
    embedding = model.encode(query_text).tolist()
    collection = _get_chroma_collection()

    where = {"domain": domain_filter} if domain_filter and domain_filter != "기타" else None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": doc,
            "score": 1 - dist,  # cosine similarity
            "rank": i + 1,
            **meta,
            "statutes": json.loads(meta.get("statutes", "[]")),
        })

    return chunks


def _bm25_search(query_text: str, top_k: int = TOP_K_BM25) -> list[dict]:
    """BM25 키워드 검색."""
    index_data = _get_bm25_index()
    bm25: BM25Okapi = index_data["bm25"]
    tokenized_query = tokenize(query_text)  # 인덱싱과 동일한 형태소 토큰화
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    chunks = []
    for rank, idx in enumerate(top_indices):
        meta = index_data["metadatas"][idx]
        chunks.append({
            "chunk_id": index_data["chunk_ids"][idx],
            "text": index_data["texts"][idx],
            "score": float(scores[idx]),
            "rank": rank + 1,
            **meta,
        })

    return chunks


def _reciprocal_rank_fusion(
    dense_results: list[dict], bm25_results: list[dict], k: int = RRF_K
) -> list[dict]:
    """RRF로 두 검색 결과 결합."""
    scores: dict[str, float] = {}
    all_chunks: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results, 1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        all_chunks[cid] = chunk

    for rank, chunk in enumerate(bm25_results, 1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        if cid not in all_chunks:
            all_chunks[cid] = chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [{"rrf_score": scores[cid], **all_chunks[cid]} for cid in sorted_ids]


def _rerank(query: str, candidates: list[dict], top_k: int = TOP_K_FINAL) -> list[RetrievedChunk]:
    """Cross-encoder(ko-reranker) 재랭킹. 실패 시 임베딩 코사인 유사도 폴백."""
    if not candidates:
        return []

    pool = candidates[:top_k * 4]

    try:
        reranker = _get_reranker()
        pairs = [(query, c["text"]) for c in pool]
        rerank_scores = reranker.predict(pairs).tolist()
    except Exception as e:
        logger.warning(f"Cross-encoder reranking 실패: {e}. 임베딩 유사도로 대체.")
        try:
            model = _get_embedding_model()
            query_emb = model.encode(query, normalize_embeddings=True)
            chunk_texts = [c["text"] for c in pool]
            chunk_embs = model.encode(chunk_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            rerank_scores = (chunk_embs @ query_emb).tolist()
        except Exception:
            logger.warning("임베딩 reranking도 실패. RRF 순서 사용.")
            rerank_scores = [c.get("rrf_score", 0) for c in pool]

    scored = sorted(zip(rerank_scores, pool), key=lambda x: x[0], reverse=True)

    return [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            case_id=c.get("case_id", ""),
            case_name=c.get("case_name", ""),
            court=c.get("court", ""),
            date=c.get("date", ""),
            domain=c.get("domain", ""),
            text=c["text"],
            statutes=c.get("statutes", []),
            source=c.get("source", ""),
            verdict=c.get("verdict", ""),
            rrf_score=c.get("rrf_score", 0.0),
            rerank_score=float(score),
        )
        for score, c in scored[:top_k]
    ]


def retrieve(
    query: str,
    hyde_document: str | None = None,
    domain: str | None = None,
    top_k: int = TOP_K_FINAL,
) -> list[RetrievedChunk]:
    """
    Hybrid retrieval 실행.
    query: 재작성된 법률 쿼리
    hyde_document: HyDE 가상 판례 문서 (있으면 dense 검색에 사용)
    domain: 법적 영역 필터
    """
    search_text = hyde_document if hyde_document else query

    logger.info(f"Dense 검색 중... (domain={domain})")
    dense_results = _dense_search(search_text, domain_filter=domain, top_k=TOP_K_DENSE)

    # 도메인 필터로 결과가 없으면 필터 없이 재시도 (오분류/데이터 편중 대비)
    if not dense_results and domain and domain != "기타":
        logger.info("도메인 필터 결과 0건. 필터 없이 재검색.")
        dense_results = _dense_search(search_text, domain_filter=None, top_k=TOP_K_DENSE)

    logger.info("BM25 검색 중...")
    bm25_results = _bm25_search(query, top_k=TOP_K_BM25)

    logger.info("RRF 결합 중...")
    fused = _reciprocal_rank_fusion(dense_results, bm25_results)

    logger.info(f"Reranking 중 (top {top_k})...")
    final = _rerank(query, fused, top_k=top_k)

    logger.info(f"검색 완료: {len(final)}개 판례 반환")
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = retrieve(
        query="금전 채무 불이행 손해배상",
        domain="민사",
        top_k=3,
    )
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.case_name} ({r.court}, {r.date})")
        print(f"     rerank_score={r.rerank_score:.4f}, rrf_score={r.rrf_score:.4f}")
        print(f"     {r.text[:100]}...")
