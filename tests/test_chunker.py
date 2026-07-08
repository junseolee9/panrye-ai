"""청커 경계/메타데이터 헤더 검증."""
from panrye.data.chunker import build_rich_text, chunk_precedent, sliding_window_chunks


def _precedent(text_len: int = 3000) -> dict:
    return {
        "case_id": "12345",
        "case_number": "2023다1111",
        "case_name": "손해배상(기)",
        "court": "대법원",
        "date": "2023.05.15",
        "domain": "민사",
        "summary": "채무불이행에 따른 손해배상 인정 여부",
        "full_text": "원고는 피고에게 금전을 대여하였다. " * (text_len // 20),
        "statutes": ["민법 제390조"],
        "verdict": "피고는 원고에게 지급하라.",
        "source": "law.go.kr",
    }


def test_chunks_respect_size_limit():
    chunks = chunk_precedent(_precedent(), chunk_size=512, overlap=64)
    assert chunks, "청크가 생성되어야 함"
    assert all(len(c.text) <= 512 for c in chunks)


def test_metadata_header_in_rich_text():
    rich = build_rich_text(_precedent())
    assert rich.startswith("사건명:")
    assert "참조법조: 민법 제390조" in rich
    assert "법원: 대법원" in rich


def test_chunk_carries_metadata():
    chunks = chunk_precedent(_precedent())
    first = chunks[0]
    assert first.case_number == "2023다1111"
    assert first.domain == "민사"
    assert first.chunk_id == "12345_chunk000"
    assert first.total_chunks == len(chunks)


def test_sliding_window_overlap():
    paragraphs = [f"문단 {i}번 내용입니다. " * 5 for i in range(10)]
    chunks = sliding_window_chunks(paragraphs, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # 인접 청크는 오버랩 텍스트를 공유
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev[-20:].strip()[:10] in nxt


def test_short_text_single_chunk():
    prec = _precedent()
    prec["full_text"] = "짧은 판결문 본문입니다. 채무자는 변제하라."
    chunks = chunk_precedent(prec)
    assert len(chunks) == 1
