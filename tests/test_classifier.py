"""질의 분류기: 정직한 confidence + LLM 폴백 경로 (Groq 목)."""
from unittest.mock import patch

from panrye.agents.classifier import _keyword_classify, classify_domain


def test_unambiguous_query_high_confidence():
    domain, conf = _keyword_classify("근로기준법상 부당해고 구제신청 절차가 궁금해요")
    assert domain == "노동"
    assert conf >= 0.7


def test_gibberish_zero_confidence():
    domain, conf = _keyword_classify("오늘 날씨가 좋네요")
    assert domain == "기타"
    assert conf == 0.0


def test_confidence_is_score_ratio():
    # confidence = top1/(top1+top2) ∈ (0.5, 1.0]
    _, conf = _keyword_classify("전세 보증금 반환 소송과 손해배상")
    assert 0.5 < conf <= 1.0


def test_keyword_path_skips_llm():
    with patch("panrye.agents.classifier._llm_classify") as mock_llm:
        result = classify_domain("음주운전 처벌 기준이 궁금합니다")
    mock_llm.assert_not_called()
    assert result["method"] == "keyword"
    assert result["domain"] == "형사"


def test_ambiguous_query_uses_llm_fallback():
    with patch("panrye.agents.classifier._llm_classify", return_value="형사") as mock_llm:
        result = classify_domain("억울한 일을 당했어요")
    mock_llm.assert_called_once()
    assert result["method"] == "llm_fallback"
    assert result["domain"] == "형사"
    assert result["confidence"] <= 0.7  # 폴백 결과에 과신 금지


def test_llm_failure_falls_back_to_gita():
    with patch("panrye.agents.classifier._llm_classify", return_value="기타"):
        result = classify_domain("도와주세요 어떻게 해야 하나요")
    assert result["domain"] == "기타"
