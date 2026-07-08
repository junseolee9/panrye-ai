"""
한국어 형태소 토크나이저 (BM25 인덱싱과 검색에서 동일하게 사용).
kiwipiepy 미설치/로드 실패 시 공백 분리로 폴백.
"""
from __future__ import annotations
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# 검색에 유의미한 품사: 명사(NNG/NNP/NNB/NR), 외국어(SL), 숫자(SN), 동사/형용사(VV/VA), 어근(XR)
_CONTENT_TAGS = ("NNG", "NNP", "NNB", "NR", "SL", "SN", "VV", "VA", "XR")


@lru_cache(maxsize=1)
def _get_kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()


def tokenize(text: str) -> list[str]:
    """형태소 기반 토큰화. 실패 시 공백 분리 폴백."""
    try:
        kiwi = _get_kiwi()
    except Exception as e:
        logger.warning(f"kiwipiepy 로드 실패: {e}. 공백 분리 사용.")
        return text.split()
    return [t.form for t in kiwi.tokenize(text) if t.tag.startswith(_CONTENT_TAGS)]
