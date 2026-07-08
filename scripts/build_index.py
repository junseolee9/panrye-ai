"""청킹 + Dense/BM25 인덱스 빌드 CLI.

사용:
    python scripts/build_index.py            # raw_data/all_precedents.json → artifacts/
    python scripts/build_index.py --fresh    # 기존 chroma 컬렉션 삭제 후 재구축
"""
from __future__ import annotations

import argparse
import logging
import shutil

from panrye.config import get_settings
from panrye.data.chunker import chunk_all_precedents, get_chunk_stats
from panrye.data.index_build import build_all_indexes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="인덱스 빌드")
    parser.add_argument("--fresh", action="store_true", help="기존 chroma_db 삭제 후 재구축")
    parser.add_argument("--skip-chunking", action="store_true", help="기존 chunks.json 사용")
    args = parser.parse_args()

    settings = get_settings()

    if args.fresh and settings.chroma_dir.exists():
        # chroma는 증분 add 방식이라 재구축 시 디렉토리 삭제가 필수
        print(f"기존 인덱스 삭제: {settings.chroma_dir}")
        shutil.rmtree(settings.chroma_dir)

    if not args.skip_chunking:
        chunks = chunk_all_precedents()
        stats = get_chunk_stats(chunks)
        print("\n청킹 통계:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    result = build_all_indexes()
    print("\n인덱싱 결과:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
