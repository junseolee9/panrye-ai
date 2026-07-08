"""
Agent 4: Summarizer
검색된 판례를 3-5문장으로 요약 + 핵심 법조항 추출.
kobart-summarization 사용 (HF free).
"""
import logging
import re
from functools import lru_cache

from transformers import pipeline

from agents.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 1024
SUMMARY_MAX_LENGTH = 150
SUMMARY_MIN_LENGTH = 30

# 청크 텍스트 앞에 붙는 메타데이터 헤더 (chunker.build_rich_text) — 요약 입력에서 제외
_META_PREFIXES = ("사건명:", "법원:", "법적 영역:", "참조법조:")


@lru_cache(maxsize=1)
def _get_summarizer():
    logger.info("요약 모델 로딩: digit82/kobart-summarization")
    return pipeline(
        "summarization",
        model="digit82/kobart-summarization",
        device=-1,
    )


def _strip_metadata_header(text: str) -> str:
    """메타데이터 헤더 라인 제거 (요약 대상은 본문만)."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if any(stripped.startswith(p) for p in _META_PREFIXES):
            continue
        if stripped.startswith("판시사항:"):
            stripped = stripped[len("판시사항:"):].strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def _cut_at_sentence(text: str, limit: int) -> str:
    """limit 이내에서 마지막 문장 경계로 절단."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_end = max(cut.rfind("."), cut.rfind("。"), cut.rfind("다."))
    if last_end > limit // 2:
        return cut[: last_end + 1]
    return cut + "..."


def summarize_chunk(text: str) -> str:
    """kobart로 판례 요약."""
    body = _strip_metadata_header(text)
    if len(body) < 200:
        return body or text[:300]

    truncated = _cut_at_sentence(body, MAX_INPUT_LENGTH)
    try:
        summarizer = _get_summarizer()
        result = summarizer(
            truncated,
            max_length=SUMMARY_MAX_LENGTH,
            min_length=SUMMARY_MIN_LENGTH,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,  # "24. 24. 24." 식 반복 생성 방지
        )
        summary = result[0]["summary_text"].strip()
        # 반복 붕괴 등으로 비정상 출력이면 원문 앞부분으로 대체
        if len(summary) < 20:
            raise ValueError(f"요약 결과 비정상: {summary!r}")
        return summary
    except Exception as e:
        logger.warning(f"요약 실패: {e}. 원문 앞부분 사용.")
        return _cut_at_sentence(body, 300)


def extract_key_statutes(statutes: list[str]) -> str:
    """법조항 목록에서 핵심 조문 추출 (최대 3개)."""
    if not statutes:
        return "명시 없음"
    clean = [s.strip() for s in statutes if s.strip()][:3]
    return ", ".join(clean) if clean else "명시 없음"


def format_verdict_snippet(verdict: str) -> str:
    """주문 전문을 한 줄로 정리 (잘라내지 않음). 표시 길이 제어는 UI 담당."""
    if not verdict:
        return ""
    flat = re.sub(r"\s+", " ", verdict).strip()
    # 수집 단계에서 300자 컷된 경우 말줄임 표시
    return flat + ("…" if len(flat) >= 300 else "")


def summarize_precedents(chunks: list[RetrievedChunk]) -> list[dict]:
    """
    검색된 판례 청크들을 요약하여 최종 컨텍스트 생성.
    동일 case_id는 첫 번째 청크만 요약 (중복 방지).
    """
    seen_cases: set[str] = set()
    summaries = []

    for chunk in chunks:
        if chunk.case_id in seen_cases:
            continue
        seen_cases.add(chunk.case_id)

        summary_text = summarize_chunk(chunk.text)
        key_statutes = extract_key_statutes(chunk.statutes)
        verdict_snippet = format_verdict_snippet(chunk.verdict)

        summaries.append({
            "case_id": chunk.case_id,
            "case_name": chunk.case_name,
            "court": chunk.court,
            "date": chunk.date,
            "domain": chunk.domain,
            "summary": summary_text,
            "key_statutes": key_statutes,
            "verdict_snippet": verdict_snippet,
            "rerank_score": chunk.rerank_score,
            "source": chunk.source,
        })

    return summaries


def format_context_for_llm(summaries: list[dict]) -> str:
    """LLM 프롬프트에 삽입할 판례 컨텍스트 포맷."""
    parts = []
    for i, s in enumerate(summaries, 1):
        part = f"""[판례 {i}]
사건명: {s['case_name']}
법원: {s['court']} | 날짜: {s['date']} | 영역: {s['domain']}
참조법조: {s['key_statutes']}
요약: {s['summary']}"""
        if s["verdict_snippet"]:
            part += f"\n주문: {s['verdict_snippet']}"
        parts.append(part)

    return "\n\n".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 더미 테스트
    dummy = RetrievedChunk(
        chunk_id="test_001",
        case_id="2023다12345",
        case_name="손해배상(기)",
        court="대법원",
        date="2023.05.15",
        domain="민사",
        text="원고는 피고에게 3,000,000원을 대여하였으나 피고가 변제 기일이 지나도록 이를 이행하지 않았다. 법원은 민법 제390조 채무불이행에 의한 손해배상 청구를 인용하였다.",
        statutes=["민법 제390조", "민법 제393조"],
        source="law.go.kr",
        verdict="피고는 원고에게 금 3,000,000원 및 이에 대한 지연손해금을 지급하라.",
        rrf_score=0.85,
        rerank_score=0.92,
    )
    results = summarize_precedents([dummy])
    print(format_context_for_llm(results))
