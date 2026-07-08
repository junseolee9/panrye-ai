"""
판례 텍스트 청킹 모듈.
전략: 문단 단위 분할 + 512자 윈도우 + 64자 오버랩 (토큰 ≈ 문자/1.5 for Korean).
각 청크 앞에 메타데이터 헤더를 붙여 검색 품질을 높인다.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from panrye.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    case_id: str
    case_number: str
    case_name: str
    court: str
    date: str
    domain: str
    text: str
    chunk_index: int
    total_chunks: int
    statutes: list[str]
    source: str
    verdict: str | None = None


def split_into_paragraphs(text: str) -> list[str]:
    """법률 문서 특성 고려한 문단 분리."""
    paragraphs = re.split(r"\n{2,}|(?<=[.。])\s*\n", text)
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


def chunk_precedent(
    precedent: dict, chunk_size: int | None = None, overlap: int | None = None
) -> list[Chunk]:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    rich_text = build_rich_text(precedent)
    paragraphs = split_into_paragraphs(rich_text)

    if not paragraphs:
        return []

    raw_chunks = sliding_window_chunks(paragraphs, chunk_size, overlap)
    total = len(raw_chunks)

    return [
        Chunk(
            chunk_id=f"{precedent['case_id']}_chunk{i:03d}",
            case_id=precedent["case_id"],
            case_number=precedent.get("case_number", ""),
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


def chunk_all_precedents(
    input_path: Path | None = None, output_path: Path | None = None
) -> list[Chunk]:
    from tqdm import tqdm

    settings = get_settings()
    input_path = input_path or settings.raw_data_dir / "all_precedents.json"
    output_path = output_path or settings.chunks_path

    if not input_path.exists():
        raise FileNotFoundError(f"원본 데이터 없음: {input_path}. ingest 먼저 실행하세요.")

    with open(input_path, encoding="utf-8") as f:
        precedents = json.load(f)

    logger.info(f"청킹 시작: {len(precedents)}건")
    all_chunks: list[Chunk] = []

    for prec in tqdm(precedents, desc="청킹"):
        all_chunks.extend(chunk_precedent(prec))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    logger.info(f"청킹 완료: {len(all_chunks)}개 청크 → {output_path}")
    return all_chunks


def get_chunk_stats(chunks: list[Chunk]) -> dict:
    from collections import Counter

    lengths = [len(c.text) for c in chunks]
    domain_dist = Counter(c.domain for c in chunks)
    return {
        "total_chunks": len(chunks),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "domain_distribution": dict(domain_dist.most_common()),
    }
