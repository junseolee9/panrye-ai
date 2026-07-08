"""
판례 텍스트 청킹 모듈.
전략: 문단 단위 분할 + 512 토큰 윈도우 + 64 토큰 오버랩.
각 청크에 메타데이터 부착 (case_id, domain, court, date).
"""
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512       # 문자 기준 (토큰 ≈ 문자/1.5 for Korean)
CHUNK_OVERLAP = 64
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
CHUNKED_DATA_DIR = PROJECT_ROOT / "chunked_data"
CHUNKED_DATA_DIR.mkdir(exist_ok=True)


@dataclass
class Chunk:
    chunk_id: str
    case_id: str
    case_name: str
    court: str
    date: str
    domain: str
    text: str
    chunk_index: int
    total_chunks: int
    statutes: list[str]
    source: str
    verdict: Optional[str] = None


def split_into_paragraphs(text: str) -> list[str]:
    """법률 문서 특성 고려한 문단 분리."""
    # 번호 목록, 줄바꿈 기준 분리
    paragraphs = re.split(r'\n{2,}|(?<=[.。])\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]


def sliding_window_chunks(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """문단 목록을 슬라이딩 윈도우로 청킹."""
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current = current + " " + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            # 오버랩: 이전 청크 끝부분 포함
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + para if overlap_text else para

            # 단일 문단이 chunk_size 초과하면 강제 분할
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size].strip())
                current = current[chunk_size - overlap:]

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_rich_text(precedent: dict) -> str:
    """검색 품질 향상을 위해 메타데이터 + 본문 결합."""
    parts = []
    if precedent.get("case_name"):
        parts.append(f"사건명: {precedent['case_name']}")
    if precedent.get("court"):
        parts.append(f"법원: {precedent['court']}")
    if precedent.get("domain"):
        parts.append(f"법적 영역: {precedent['domain']}")
    if precedent.get("statutes"):
        statutes = precedent["statutes"]
        if isinstance(statutes, list):
            parts.append(f"참조법조: {', '.join(statutes[:3])}")
    if precedent.get("summary"):
        parts.append(f"판시사항: {precedent['summary']}")
    if precedent.get("full_text"):
        parts.append(precedent["full_text"])
    return "\n".join(parts)


def chunk_precedent(precedent: dict) -> list[Chunk]:
    rich_text = build_rich_text(precedent)
    paragraphs = split_into_paragraphs(rich_text)

    if not paragraphs:
        return []

    raw_chunks = sliding_window_chunks(paragraphs, CHUNK_SIZE, CHUNK_OVERLAP)
    total = len(raw_chunks)

    return [
        Chunk(
            chunk_id=f"{precedent['case_id']}_chunk{i:03d}",
            case_id=precedent["case_id"],
            case_name=precedent.get("case_name", ""),
            court=precedent.get("court", ""),
            date=precedent.get("date", ""),
            domain=precedent.get("domain", "기타"),
            text=text,
            chunk_index=i,
            total_chunks=total,
            statutes=precedent.get("statutes", []),
            source=precedent.get("source", ""),
            verdict=precedent.get("verdict", ""),
        )
        for i, text in enumerate(raw_chunks)
    ]


def chunk_all_precedents(input_file: str = "all_precedents.json") -> list[Chunk]:
    input_path = RAW_DATA_DIR / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"원본 데이터 없음: {input_path}. ingest.py 먼저 실행하세요.")

    with open(input_path, encoding="utf-8") as f:
        precedents = json.load(f)

    logger.info(f"청킹 시작: {len(precedents)}건")
    all_chunks: list[Chunk] = []

    for prec in tqdm(precedents, desc="청킹"):
        chunks = chunk_precedent(prec)
        all_chunks.extend(chunks)

    out_path = CHUNKED_DATA_DIR / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    logger.info(f"청킹 완료: {len(all_chunks)}개 청크 → {out_path}")
    return all_chunks


def get_chunk_stats(chunks: list[Chunk]) -> dict:
    lengths = [len(c.text) for c in chunks]
    from collections import Counter
    domain_dist = Counter(c.domain for c in chunks)
    return {
        "total_chunks": len(chunks),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "domain_distribution": dict(domain_dist.most_common()),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunks = chunk_all_precedents()
    stats = get_chunk_stats(chunks)
    print("\n청킹 통계:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
