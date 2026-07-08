"""코퍼스 분포 통계 — README용 마크다운 표 출력.

사용: python scripts/corpus_stats.py [--chunks]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from panrye.config import get_settings


def _decade(date: str) -> str:
    year = date[:4]
    return f"{year[:3]}0년대" if len(year) == 4 and year.isdigit() else "미상"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", action="store_true", help="청크 단위 분포도 출력")
    args = parser.parse_args()

    settings = get_settings()
    with open(settings.raw_data_dir / "all_precedents.json", encoding="utf-8") as f:
        precedents = json.load(f)

    total = len(precedents)
    domain_dist = Counter(p["domain"] for p in precedents)
    court_dist = Counter(p["court"] or "미상" for p in precedents)
    decade_dist = Counter(_decade(p["date"]) for p in precedents)
    with_meta = sum(1 for p in precedents if p["court"] and p["date"] and p["case_number"])

    print(f"## 코퍼스 통계 (판례 {total:,}건)\n")
    print(f"법원·선고일·사건번호 완비율: {with_meta / total * 100:.1f}%\n")

    print("| 도메인 | 판례 수 | 비율 |")
    print("|---|---|---|")
    for d, c in domain_dist.most_common():
        print(f"| {d} | {c:,} | {c / total * 100:.1f}% |")

    print("\n| 법원 | 판례 수 |")
    print("|---|---|")
    for court, c in court_dist.most_common(8):
        print(f"| {court} | {c:,} |")

    print("\n| 연대 | 판례 수 |")
    print("|---|---|")
    for dec, c in sorted(decade_dist.items()):
        print(f"| {dec} | {c:,} |")

    if args.chunks and settings.chunks_path.exists():
        with open(settings.chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        chunk_dist = Counter(c["domain"] for c in chunks)
        print(f"\n| 도메인 | 청크 수 | 비율 |  (총 {len(chunks):,}개)")
        print("|---|---|---|")
        for d, c in chunk_dist.most_common():
            print(f"| {d} | {c:,} | {c / len(chunks) * 100:.1f}% |")


if __name__ == "__main__":
    main()
