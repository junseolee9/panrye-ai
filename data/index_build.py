"""
ChromaDB 인덱싱 모듈.
- Dense: ko-sroberta-multitask 임베딩
- BM25 인덱스도 병렬 구축 (retriever.py에서 사용)
- 배치 처리로 메모리 효율화
"""
import json
import logging
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from dotenv import load_dotenv

from data.tokenizer import tokenize

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKED_DATA_DIR = PROJECT_ROOT / "chunked_data"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
BM25_INDEX_PATH = PROJECT_ROOT / "bm25_index.pkl"

EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
COLLECTION_NAME = "panrye_precedents"
BATCH_SIZE = 128


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DB_PATH.mkdir(exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_chunks(input_file: str = "chunks.json") -> list[dict]:
    path = CHUNKED_DATA_DIR / input_file
    if not path.exists():
        raise FileNotFoundError(f"청크 데이터 없음: {path}. chunker.py 먼저 실행하세요.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dense_index(chunks: list[dict], collection: chromadb.Collection) -> None:
    """ko-sroberta 임베딩으로 ChromaDB 인덱스 구축."""
    logger.info(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

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
            metadatas=[
                {
                    "case_id": c["case_id"],
                    "case_name": c["case_name"],
                    "court": c["court"],
                    "date": c["date"],
                    "domain": c["domain"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "statutes": json.dumps(c["statutes"], ensure_ascii=False),
                    "source": c["source"],
                    "verdict": c.get("verdict", "") or "",
                }
                for c in batch
            ],
        )

    logger.info(f"Dense 인덱싱 완료: {collection.count()}개 벡터 저장됨")


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """BM25 역인덱스 구축 및 pickle 저장 (형태소 토큰화)."""
    logger.info("BM25 인덱스 구축 중 (kiwi 형태소 토큰화)...")
    tokenized = [tokenize(c["text"]) for c in tqdm(chunks, desc="BM25 토큰화")]
    bm25 = BM25Okapi(tokenized)

    index_data = {
        "bm25": bm25,
        "tokenizer": "kiwi",
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metadatas": [
            {
                "case_id": c["case_id"],
                "case_name": c["case_name"],
                "court": c["court"],
                "date": c["date"],
                "domain": c["domain"],
                "statutes": c["statutes"],
                "source": c["source"],
                "verdict": c.get("verdict", "") or "",
            }
            for c in chunks
        ],
    }

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)

    logger.info(f"BM25 인덱스 저장: {BM25_INDEX_PATH}")
    return bm25


def build_all_indexes() -> dict:
    chunks = load_chunks()
    logger.info(f"청크 로드 완료: {len(chunks)}개")

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    build_dense_index(chunks, collection)
    build_bm25_index(chunks)

    return {
        "total_chunks": len(chunks),
        "chroma_count": collection.count(),
        "bm25_index_path": str(BM25_INDEX_PATH),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = build_all_indexes()
    print("\n인덱싱 결과:")
    for k, v in result.items():
        print(f"  {k}: {v}")
