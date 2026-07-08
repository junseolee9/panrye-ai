"""
Agent 1: Domain Classifier
가중 키워드 스코어링 (core.domains 렉시콘) + 애매하면 Groq 1콜 폴백.
confidence는 실제 점수 비율 — top1 / (top1 + top2).
"""
from __future__ import annotations

import logging

from panrye.config import get_settings
from panrye.core.domains import DOMAINS, FALLBACK_DOMAIN, score_domains

logger = logging.getLogger(__name__)


def _keyword_classify(query: str) -> tuple[str, float]:
    """가중 렉시콘 스코어링. Returns (domain, confidence)."""
    scores = score_domains(query)
    if not scores:
        return FALLBACK_DOMAIN, 0.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1_domain, top1 = ranked[0]
    top2 = ranked[1][1] if len(ranked) > 1 else 0
    confidence = top1 / (top1 + top2) if (top1 + top2) > 0 else 0.0
    return top1_domain, round(confidence, 3)


def _llm_classify(query: str) -> str:
    """Groq 제약 출력 분류. 실패/비정상 출력 시 기타."""
    from groq import Groq

    settings = get_settings()
    if not settings.groq_api_key:
        return FALLBACK_DOMAIN

    prompt = (
        "다음 법률 상담 질문이 속하는 영역을 아래 목록에서 정확히 하나만 골라 "
        "그 단어만 출력하세요.\n"
        f"목록: {' / '.join(DOMAINS)} / {FALLBACK_DOMAIN}\n\n"
        f"질문: {query}"
    )
    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.fast_llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM 분류 실패: {e}. {FALLBACK_DOMAIN} 처리.")
        return FALLBACK_DOMAIN

    # 방어적 파싱: 출력에 포함된 첫 도메인 채택
    for d in DOMAINS:
        if d in raw:
            return d
    return FALLBACK_DOMAIN


def classify_domain(query: str) -> dict:
    """
    도메인 분류. 키워드 점수가 확실하면 그대로, 애매하면 LLM 폴백.
    Returns: {domain, confidence, method}
    """
    threshold = get_settings().classifier_confidence_threshold
    domain, confidence = _keyword_classify(query)

    if domain != FALLBACK_DOMAIN and confidence >= threshold:
        logger.info(f"키워드 분류: {domain} (confidence={confidence})")
        return {"domain": domain, "confidence": confidence, "method": "keyword"}

    llm_domain = _llm_classify(query)
    # LLM 결과와 키워드 1위가 일치하면 그 합의를 신뢰
    if llm_domain == domain and domain != FALLBACK_DOMAIN:
        confidence = max(confidence, 0.7)
    elif llm_domain != FALLBACK_DOMAIN:
        domain, confidence = llm_domain, 0.5
    logger.info(f"LLM 폴백 분류: {domain} (confidence={confidence})")
    return {"domain": domain, "confidence": confidence, "method": "llm_fallback"}
