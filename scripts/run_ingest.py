"""판례 수집 CLI.

사용:
    python scripts/run_ingest.py --all                     # 기본 목표치로 전 도메인
    python scripts/run_ingest.py --domain 부동산 --target 800
중단 후 재실행하면 이어서 수집한다 (raw_data/ingest_state.json).
"""
from __future__ import annotations

import argparse
import logging

from panrye.data.ingest import DEFAULT_TARGETS, collect_all, collect_domain


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="법제처 판례 수집")
    parser.add_argument("--all", action="store_true", help="전 도메인 기본 목표치 수집")
    parser.add_argument("--domain", type=str, help="단일 도메인")
    parser.add_argument("--target", type=int, default=None, help="목표 건수")
    args = parser.parse_args()

    if args.all:
        store = collect_all()
    elif args.domain:
        target = args.target or DEFAULT_TARGETS.get(args.domain, 500)
        store = collect_domain(args.domain, target)
        store.export_all()
    else:
        parser.error("--all 또는 --domain 필요")

    dist = store.domain_counts()
    print(f"\n총 {sum(dist.values())}건")
    for d, c in dist.most_common():
        print(f"  {d}: {c}건")


if __name__ == "__main__":
    main()
