"""
인덱스 구축 모듈.
- Dense: ko-sroberta-multitask 임베딩 → ChromaDB (cosine HNSW)
- Sparse: BM25 (kiwi 형태소 토큰화) → pickle
- 배치 처리 + 증분 인덱싱 (기존 id 스킵)
"""
from __future__ import annotations

import json
import logging
import pickle

from panrye.config import get_settings
from panrye.data.tokenizer import tokenize

logger = logging.getLogger(__name__)

BATCH_SIZE = 128


def get_chroma_client():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(client):
    return client.get_or_create_collection(
        name=get_settings().collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def load_chunks() -> list[dict]:
    path = get_settings().chunks_path
    if not path.exists():
        raise FileNotFoundError(f"청크 데이터 없음: {path}. chunker 먼저 실행하세요.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _chunk_metadata(c: dict, json_statutes: bool) -> dict:
    meta = {
        "case_id": c["case_id"],
        "case_name": c["case_name"],
        "court": c["court"],
        "date": c["date"],
        "domain": c["domain"],
        "statutes": (
            json.dumps(c["statutes"], ensure_ascii=False) if json_statutes else c["statutes"]
        ),
        "source": c["source"],
        "verdict": c.get("verdict", "") or "",
    }
    if json_statutes:
        meta["chunk_index"] = c["chunk_index"]
        meta["total_chunks"] = c["total_chunks"]
    return meta


def build_dense_index(chunks: list[dict], collection) -> None:
    """ko-sroberta 임베딩으로 ChromaDB 인덱스 구축."""
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    settings = get_settings()
    logger.info(f"임베딩 모델 로딩: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)

    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info("이미 인덱싱된 데이터. 건너뜀.")
        return

    logger.info(f"인덱싱 시작: {len(new_chunks)}개 청크 (배치 크기={BATCH_SIZE})")

    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Dense 인덱싱"):
        batch = new_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[_chunk_metadata(c, json_statutes=True) for c in batch],
        )

    logger.info(f"Dense 인덱싱 완료: {collection.count()}개 벡터 저장됨")


def build_bm25_index(chunks: list[dict]):
    """BM25 역인덱스 구축 및 pickle 저장 (형태소 토큰화)."""
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm

    settings = get_settings()
    logger.info("BM25 인덱스 구축 중 (kiwi 형태소 토큰화)...")
    tokenized = [tokenize(c["text"]) for c in tqdm(chunks, desc="BM25 토큰화")]
    bm25 = BM25Okapi(tokenized)

    index_data = {
        "bm25": bm25,
        "tokenizer": "kiwi",
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metadatas": [_chunk_metadata(c, json_statutes=False) for c in chunks],
    }

    settings.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.bm25_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info(f"BM25 인덱스 저장: {settings.bm25_path}")
    return bm25


def build_all_indexes() -> dict:
    settings = get_settings()
    chunks = load_chunks()
    logger.info(f"청크 로드 완료: {len(chunks)}개")

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    build_dense_index(chunks, collection)
    build_bm25_index(chunks)

    return {
        "total_chunks": len(chunks),
        "chroma_count": collection.count(),
        "bm25_index_path": str(settings.bm25_path),
    }
