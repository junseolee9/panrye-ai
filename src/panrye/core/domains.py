"""
법률 도메인 렉시콘 — 단일 소스.
수집 단계 라벨링(classify_precedent)과 질의 시점 분류(agents.classifier)가 함께 사용한다.
라벨 체계: 형사/민사/가족법/행정/노동/부동산/기타
우선순위(가중치): 참조법령(4) > 민법 가족힌트(2) = 사건종류명(2) > 본문 키워드(1)
"""
from __future__ import annotations

DOMAINS = ["형사", "민사", "가족법", "행정", "노동", "부동산"]
FALLBACK_DOMAIN = "기타"

# (법령명 substring, 도메인). 법령명이 가장 신뢰도 높은 신호.
LAW_TO_DOMAIN: list[tuple[str, str]] = [
    ("근로기준법", "노동"), ("노동조합", "노동"), ("산업재해", "노동"),
    ("최저임금", "노동"), ("퇴직급여", "노동"), ("산업안전", "노동"),
    ("기간제", "노동"), ("파견근로", "노동"), ("임금채권", "노동"),
    ("주택임대차보호법", "부동산"), ("상가건물임대차보호법", "부동산"),
    ("상가건물 임대차보호법", "부동산"), ("부동산등기", "부동산"),
    ("공인중개사", "부동산"), ("집합건물", "부동산"), ("부동산 실권리자", "부동산"),
    ("가사소송법", "가족법"), ("가족관계의 등록", "가족법"),
    ("형법", "형사"), ("형사소송법", "형사"), ("도로교통법", "형사"),
    ("특정범죄", "형사"), ("폭력행위", "형사"), ("성폭력", "형사"),
    ("아동·청소년", "형사"), ("아동ㆍ청소년", "형사"), ("마약류", "형사"),
    ("교통사고처리", "형사"), ("특정경제범죄", "형사"), ("경범죄", "형사"),
    ("스토킹", "형사"), ("전자금융거래법", "형사"), ("정보통신망 이용촉진", "형사"),
    ("행정소송법", "행정"), ("행정절차법", "행정"), ("행정심판법", "행정"),
    ("국세", "행정"), ("지방세", "행정"), ("소득세법", "행정"),
    ("부가가치세", "행정"), ("법인세", "행정"), ("국가배상법", "행정"),
    ("출입국관리", "행정"), ("식품위생법", "행정"), ("국민건강보험", "행정"),
    ("민사소송법", "민사"), ("상법", "민사"), ("자동차손해배상", "민사"),
    ("보험업법", "민사"), ("이자제한법", "민사"), ("채무자 회생", "민사"),
    ("민사집행법", "민사"),
]

# 민법은 편별로 갈림: 친족/상속 관련 문맥이면 가족법, 아니면 민사
MINBEOP_FAMILY_HINTS = [
    "이혼", "혼인", "친권", "양육", "상속", "유류분", "유언",
    "재산분할", "위자료", "입양", "친생자", "사실혼",
]

TEXT_KEYWORDS: dict[str, list[str]] = {
    "노동": ["부당해고", "해고", "임금체불", "퇴직금", "근로자", "근로계약",
             "산재", "직장 내 괴롭힘", "연차", "휴업수당", "노동위원회",
             "통상임금", "부당노동행위", "취업규칙", "직위해제"],
    "부동산": ["임대차", "전세금", "보증금", "임차인", "임대인", "명도",
               "매매계약", "분양", "소유권이전등기", "경매", "전세보증금",
               "월세", "재개발", "재건축", "유치권", "근저당", "가등기"],
    "가족법": ["이혼", "위자료", "재산분할", "양육권", "친권", "상속",
               "유류분", "혼인", "사실혼", "유언", "면접교섭", "양육비"],
    "형사": ["절도", "사기", "폭행", "상해", "살인", "강도", "횡령", "배임",
             "음주운전", "무면허", "마약", "성폭행", "협박", "명예훼손",
             "피고인", "징역", "벌금", "기소", "약식명령", "보이스피싱"],
    "행정": ["행정처분", "과세처분", "취소소송", "영업정지", "면허취소",
             "과태료", "부과처분", "국가배상", "재심판정", "인허가"],
    "민사": ["손해배상", "대여금", "채무", "채권", "계약", "불법행위",
             "보증", "양수금", "구상금", "부당이득", "차용증"],
}

# law.go.kr 사건종류명 → 도메인
CASE_TYPE_TO_DOMAIN = {
    "형사": "형사",
    "민사": "민사",
    "가사": "가족법",
    "일반행정": "행정",
    "세무": "행정",
}

WEIGHT_LAW = 4
WEIGHT_MINBEOP = 2
WEIGHT_CASE_TYPE = 2
WEIGHT_TEXT_KEYWORD = 1


def score_domains(
    text: str,
    statutes: list[str] | None = None,
    case_type: str | None = None,
) -> dict[str, int]:
    """도메인별 가중 점수 계산. 라벨링·질의 분류 공통 코어."""
    joined_laws = " ".join(statutes) if statutes else ""
    scores: dict[str, int] = {}

    for law, domain in LAW_TO_DOMAIN:
        if law in joined_laws or law in text:
            scores[domain] = scores.get(domain, 0) + WEIGHT_LAW

    if "민법" in joined_laws or "민법" in text:
        is_family = any(h in text or h in joined_laws for h in MINBEOP_FAMILY_HINTS)
        d = "가족법" if is_family else "민사"
        scores[d] = scores.get(d, 0) + WEIGHT_MINBEOP

    if case_type and case_type in CASE_TYPE_TO_DOMAIN:
        d = CASE_TYPE_TO_DOMAIN[case_type]
        scores[d] = scores.get(d, 0) + WEIGHT_CASE_TYPE

    for domain, keywords in TEXT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits:
            scores[domain] = scores.get(domain, 0) + hits * WEIGHT_TEXT_KEYWORD

    return scores


def classify_precedent(
    statutes: list[str],
    text: str = "",
    case_name: str = "",
    case_type: str | None = None,
    default: str = FALLBACK_DOMAIN,
) -> str:
    """수집 단계 판례 라벨링: 최고 점수 도메인."""
    combined = f"{case_name} {text[:1000]}"
    scores = score_domains(combined, statutes=statutes, case_type=case_type)
    if not scores:
        return default
    return max(scores.items(), key=lambda x: x[1])[0]
