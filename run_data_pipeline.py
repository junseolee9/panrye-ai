"""
Month 1 데이터 파이프라인 실행 스크립트.
순서: 수집 → 청킹 → 인덱싱
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    from dotenv import load_dotenv
    import os
    load_dotenv()

    use_law_api = bool(os.getenv("LAW_API_KEY"))
    if not use_law_api:
        logger.warning("LAW_API_KEY 없음. KLAID 데이터셋만 사용합니다.")

    # Step 1: 수집
    logger.info("=== Step 1: 판례 수집 ===")
    from data.ingest import ingest_all
    precedents = ingest_all(use_law_api=use_law_api, use_klaid=True, max_per_keyword=30)
    logger.info(f"수집 완료: {len(precedents)}건")

    # Step 2: 청킹
    logger.info("=== Step 2: 청킹 ===")
    from data.chunker import chunk_all_precedents, get_chunk_stats
    chunks = chunk_all_precedents()
    stats = get_chunk_stats(chunks)
    logger.info(f"청킹 완료: {stats['total_chunks']}개 청크")

    # Step 3: 인덱싱
    logger.info("=== Step 3: ChromaDB + BM25 인덱싱 ===")
    from data.index_build import build_all_indexes
    result = build_all_indexes()
    logger.info(f"인덱싱 완료: ChromaDB {result['chroma_count']}개 벡터")

    print("\n데이터 파이프라인 완료!")
    print(f"  판례: {len(precedents)}건")
    print(f"  청크: {stats['total_chunks']}개")
    print(f"  벡터: {result['chroma_count']}개")
    print("\n다음 단계: python -m uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
