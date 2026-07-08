"""
Agent 1: Intent Classifier
Zero-shot classification으로 법적 영역 분류 (형사/민사/가족/행정/노동/부동산/기타).
HuggingFace pipeline 사용 (free).
"""
import logging
from functools import lru_cache

from transformers import pipeline

logger = logging.getLogger(__name__)

LEGAL_DOMAINS = ["형사", "민사", "가족법", "행정", "노동", "부동산", "기타"]

DOMAIN_KEYWORDS = {
    # 더 구체적인 도메인을 먼저 체크 (substring 충돌 방지)
    "부동산": ["전세금", "전세", "임대차", "보증금", "집주인", "세입자", "매매",
               "등기", "건축", "분양", "토지", "아파트", "월세", "부동산",
               "임대차계약", "전세계약"],
    "노동": ["부당해고", "근로계약서", "근로계약", "최저임금", "임금체불", "해고", "임금",
             "퇴직금", "산재", "직장", "고용", "월급", "급여", "알바", "아르바이트",
             "직원", "사장", "퇴직", "휴가", "야근", "주휴", "근로자"],
    "가족법": ["이혼", "상속", "친권", "양육권", "위자료", "재산분할", "유언",
               "남편", "아내", "배우자", "양육", "별거", "혼인", "결혼", "부부",
               "유산", "상속인"],
    "형사": ["절도", "사기", "폭행", "상해", "살인", "강도", "횡령", "배임",
             "마약", "강간", "협박", "명예훼손", "훔", "때렸", "때리", "맞았",
             "고소", "고발", "범죄", "처벌"],
    "행정": ["행정처분", "행정소송", "과세", "납세", "조세", "세금", "과태료",
             "면허취소", "면허정지", "면허", "허가취소", "처분", "공무원", "행정", "국가배상"],
    "민사": ["손해배상", "채권", "채무", "불법행위", "물권", "담보", "연대보증",
             "빌려", "빌려줬", "빌렸", "안 갚", "못 받", "합의", "대여", "차용",
             "환불", "배상", "청구", "소송", "계약"],
}


@lru_cache(maxsize=1)
def _get_classifier():
    logger.info("Zero-shot classifier 로딩 중 (첫 호출 시 다운로드)...")
    return pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        device=-1,
    )


def classify_keyword(text: str) -> str | None:
    """키워드 매칭으로 빠른 1차 분류 (모델 호출 없음)."""
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return domain
    return None


def classify_domain(text: str, use_model: bool = True) -> dict:
    """
    텍스트 기반 법적 영역 분류.
    1단계: 키워드 매칭 (빠름)
    2단계: Zero-shot classification (정확함)
    """
    # 1단계: 키워드 매칭
    keyword_result = classify_keyword(text)

    if not use_model or keyword_result:
        domain = keyword_result or "기타"
        return {
            "domain": domain,
            "confidence": 0.85 if keyword_result else 0.3,
            "method": "keyword",
            "all_scores": {d: (1.0 if d == domain else 0.0) for d in LEGAL_DOMAINS},
        }

    # 2단계: Zero-shot classification
    try:
        clf = _get_classifier()
        hypothesis = "이 상황은 {}과 관련된 법률 문제이다."
        result = clf(text[:512], LEGAL_DOMAINS, hypothesis_template=hypothesis, multi_label=False)

        scores = dict(zip(result["labels"], result["scores"]))
        top_domain = result["labels"][0]

        return {
            "domain": top_domain,
            "confidence": result["scores"][0],
            "method": "zero-shot",
            "all_scores": scores,
        }
    except Exception as e:
        logger.warning(f"Zero-shot 분류 실패: {e}. 기타로 처리.")
        return {
            "domain": "기타",
            "confidence": 0.0,
            "method": "fallback",
            "all_scores": {},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        "친구한테 100만원 빌려줬는데 안 갚아요",
        "남편이 바람을 피워서 이혼하고 싶어요",
        "회사에서 갑자기 해고 통보를 받았어요",
        "편의점에서 과자를 훔치다 잡혔어요",
        "아파트 전세금을 못 돌려받고 있어요",
    ]
    for t in tests:
        result = classify_domain(t, use_model=False)
        print(f"입력: {t[:30]}... → {result['domain']} ({result['confidence']:.2f}, {result['method']})")
