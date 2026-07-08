"""core.domains 가중 스코어링 검증."""
from panrye.core.domains import (
    WEIGHT_LAW,
    classify_precedent,
    score_domains,
)


def test_law_name_beats_text_keyword():
    # 본문에 민사 키워드가 있어도 근로기준법 인용이 우선해야 함
    scores = score_domains(
        "계약 위반으로 손해배상을 구함",
        statutes=["근로기준법 제23조"],
    )
    assert scores["노동"] >= WEIGHT_LAW
    assert max(scores, key=scores.get) == "노동"


def test_minbeop_family_disambiguation():
    # 민법 + 이혼 문맥 → 가족법
    domain = classify_precedent(["민법 제840조"], text="원고와 피고는 이혼 및 위자료를 다툰다")
    assert domain == "가족법"


def test_minbeop_realestate_disambiguation():
    # 민법 + 임대차 문맥 → 부동산
    domain = classify_precedent(["민법 제618조"], text="임차인이 보증금 반환을 구하는 사안")
    assert domain == "부동산"


def test_case_type_signal():
    assert classify_precedent([], text="", case_type="가사") == "가족법"


def test_empty_input_falls_back():
    assert classify_precedent([], text="") == "기타"


def test_bojeung_substring_no_collision():
    # "보증금"이 민사 "연대보증"으로 오인되지 않아야 함
    scores = score_domains("집주인이 전세 보증금을 돌려주지 않아요")
    assert max(scores, key=scores.get) == "부동산"
