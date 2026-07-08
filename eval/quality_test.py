"""
답변 품질 자동 검증 스크립트.
각 테스트 케이스에 대해:
1. 도메인 분류 정확도
2. 검색 판례 관련성 (키워드 매칭)
3. 답변 충실도 (법조문 언급, 길이, 면책고지)
4. 전반적 품질 점수 산출
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.pipeline import run_pipeline

TEST_CASES = [
    {
        "id": 1,
        "query": "친구한테 300만원 빌려줬는데 1년째 안 갚고 연락도 안 됩니다",
        "expected_domain": "민사",
        "relevant_keywords": ["채무", "채권", "대여", "불이행", "청구", "소액", "지급명령"],
        "expected_laws": ["민법"],
        "category": "금전대차",
    },
    {
        "id": 2,
        "query": "남편이 외도를 해서 이혼하고 싶습니다. 위자료를 받을 수 있나요?",
        "expected_domain": "가족법",
        "relevant_keywords": ["이혼", "위자료", "부정행위", "혼인", "재산분할"],
        "expected_laws": ["민법 제840조"],
        "category": "이혼",
    },
    {
        "id": 3,
        "query": "갑자기 회사에서 해고 통보를 받았어요. 부당해고 아닌가요?",
        "expected_domain": "노동",
        "relevant_keywords": ["해고", "부당", "근로기준법", "노동위원회", "구제"],
        "expected_laws": ["근로기준법"],
        "category": "부당해고",
    },
    {
        "id": 4,
        "query": "편의점에서 과자를 훔치다 잡혔어요. 어떻게 되나요?",
        "expected_domain": "형사",
        "relevant_keywords": ["절도", "형법", "처벌", "기소", "합의"],
        "expected_laws": ["형법"],
        "category": "절도",
    },
    {
        "id": 5,
        "query": "전세 계약이 끝났는데 집주인이 전세금 2억을 못 돌려준다고 합니다",
        "expected_domain": "부동산",
        "relevant_keywords": ["전세", "보증금", "임차", "반환", "임차권"],
        "expected_laws": ["주택임대차보호법"],
        "category": "전세금반환",
    },
]


def score_answer(result: dict, test_case: dict) -> dict:
    scores = {}
    notes = []

    # 1. 도메인 분류 정확도
    domain_ok = result.get("domain") == test_case["expected_domain"]
    scores["domain_correct"] = 1.0 if domain_ok else 0.0
    if not domain_ok:
        notes.append(f"도메인 오류: {result.get('domain')} (예상: {test_case['expected_domain']})")

    # 2. 검색 판례 관련성
    retrieved = result.get("retrieved_chunks", [])
    answer = result.get("final_answer", "")
    context = " ".join(s.get("summary", "") for s in result.get("summaries", []))
    all_text = answer + " " + context

    kw_hits = sum(1 for kw in test_case["relevant_keywords"] if kw in all_text)
    kw_score = kw_hits / len(test_case["relevant_keywords"])
    scores["keyword_relevance"] = round(kw_score, 2)
    notes.append(f"키워드 매칭: {kw_hits}/{len(test_case['relevant_keywords'])} ({kw_score:.0%})")

    # 3. 법조문 언급
    law_hits = sum(1 for law in test_case["expected_laws"] if law in answer)
    scores["law_citation"] = 1.0 if law_hits > 0 else 0.0
    if law_hits == 0:
        notes.append(f"법조문 미언급: {test_case['expected_laws']}")

    # 4. 답변 길이 (충실도)
    answer_len = len(answer.replace("---", "").replace("⚠️", ""))
    scores["answer_length"] = min(answer_len / 300, 1.0)
    notes.append(f"답변 길이: {answer_len}자")

    # 5. 면책 고지 포함
    disclaimer_ok = "면책" in answer or "법률 자문" in answer or "전문가" in answer
    scores["has_disclaimer"] = 1.0 if disclaimer_ok else 0.0
    if not disclaimer_ok:
        notes.append("면책 고지 없음")

    # 6. 판례 검색 결과 수
    scores["retrieved_count"] = min(len(retrieved) / 3, 1.0)
    notes.append(f"검색된 판례: {len(retrieved)}건")

    # 종합 점수
    weights = {
        "domain_correct": 0.25,
        "keyword_relevance": 0.30,
        "law_citation": 0.20,
        "answer_length": 0.10,
        "has_disclaimer": 0.10,
        "retrieved_count": 0.05,
    }
    total = sum(scores[k] * w for k, w in weights.items())
    scores["total"] = round(total, 3)

    return {"scores": scores, "notes": notes}


def run_quality_test():
    print("=" * 60)
    print("PanRye.ai 품질 검증 테스트")
    print("=" * 60)

    results = []

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['category']}: {tc['query'][:45]}...")
        start = time.time()

        try:
            result = run_pipeline(tc["query"])
            latency = time.time() - start

            score_result = score_answer(result, tc)
            scores = score_result["scores"]
            notes = score_result["notes"]

            print(f"  도메인: {result.get('domain')} | 재작성: {result.get('rewritten_query', '')[:40]}")
            print(f"  레이턴시: {latency:.1f}초")
            for note in notes:
                print(f"  • {note}")
            print(f"  ★ 종합 점수: {scores['total']:.2f}/1.00")
            print(f"  답변 미리보기: {result.get('final_answer','')[:120]}...")

            results.append({
                "id": tc["id"],
                "category": tc["category"],
                "latency": latency,
                **scores,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": tc["id"], "category": tc["category"], "total": 0.0, "error": str(e)})

    # 종합 리포트
    print("\n" + "=" * 60)
    print("종합 결과")
    print("=" * 60)
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_total = sum(r["total"] for r in valid) / len(valid)
        avg_latency = sum(r["latency"] for r in valid) / len(valid)
        avg_domain = sum(r.get("domain_correct", 0) for r in valid) / len(valid)
        avg_kw = sum(r.get("keyword_relevance", 0) for r in valid) / len(valid)
        avg_law = sum(r.get("law_citation", 0) for r in valid) / len(valid)

        print(f"  도메인 정확도:    {avg_domain:.0%}")
        print(f"  키워드 관련성:    {avg_kw:.0%}")
        print(f"  법조문 인용:      {avg_law:.0%}")
        print(f"  평균 레이턴시:    {avg_latency:.1f}초")
        print(f"  ★ 평균 종합점수:  {avg_total:.2f}/1.00")

        if avg_total >= 0.75:
            print("\n  판정: GOOD — 포트폴리오 수준 도달")
        elif avg_total >= 0.55:
            print("\n  판정: FAIR — 추가 개선 필요")
        else:
            print("\n  판정: POOR — 데이터/프롬프트 재검토 필요")

    return results


if __name__ == "__main__":
    run_quality_test()
