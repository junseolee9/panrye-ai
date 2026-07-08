"""
Agent 2: Query Reformulator
구어체 → 법률 용어 변환 + HyDE (Hypothetical Document Embedding).
HyDE: 가상의 판례 요약문 생성 후 임베딩 → 검색 정확도 향상.
"""
import logging
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def _get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY 미설정. .env 확인하세요.")
    return Groq(api_key=GROQ_API_KEY)


REWRITE_SYSTEM_PROMPT = """당신은 한국 법률 전문가입니다.
사용자의 구어체 상황 설명을 판례 검색에 최적화된 법률 용어 쿼리로 변환하세요.

변환 규칙:
1. 구어체 → 법률 용어 (예: "돈 안 갚음" → "금전채무 불이행", "해고 통보" → "부당해고 해고예고")
2. 반드시 관련 법령명 포함 (예: 민법, 근로기준법, 형법, 주택임대차보호법)
3. 핵심 법적 쟁점 + 법령명 조합으로 작성
4. 형식: "[법적쟁점] [관련법령] [세부쟁점]"
5. 50자 이내, 쿼리만 출력 (설명 없이)

예시:
- "친구가 돈을 안 갚아요" → "금전채무 불이행 민법 손해배상 청구"
- "갑자기 해고됐어요" → "부당해고 근로기준법 해고예고 노동위원회"
- "편의점에서 물건 훔쳤어요" → "절도죄 형법 제329조 초범 처벌"
- "전세금 못 받아요" → "전세금 반환 주택임대차보호법 임차권등기"""

HYDE_SYSTEM_PROMPT = """당신은 한국 법원 판결문 전문가입니다.
주어진 법률 상황에 대해 실제 대법원/지방법원 판결문처럼 보이는 가상의 판시사항을 작성하세요.

형식: "피고인/피고는 [구체적 사실관계]를 하였다. 법원은 [적용 법조문: 구체적 조문번호]에 의거하여 [법적 판단]을 하였다."
- 반드시 구체적 법조문 번호 포함 (예: 민법 제390조, 형법 제329조)
- 한국 법원 문체 사용
- 200자 이내"""


def rewrite_query(user_query: str, domain: str) -> str:
    """구어체 쿼리를 법률 용어 쿼리로 변환."""
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"[법적 영역: {domain}]\n{user_query}"},
            ],
            max_tokens=100,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"쿼리 재작성 실패: {e}. 원본 쿼리 사용.")
        return user_query


def generate_hyde_document(user_query: str, domain: str) -> str:
    """HyDE: 가상의 판례 요약문 생성."""
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"[법적 영역: {domain}]\n상황: {user_query}"},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"HyDE 생성 실패: {e}. 재작성 쿼리로 대체.")
        return user_query


def reformulate(user_query: str, domain: str) -> dict:
    """
    쿼리 재작성 + HyDE 문서 생성.
    Returns: {rewritten_query, hyde_document, original_query}
    """
    rewritten = rewrite_query(user_query, domain)
    hyde_doc = generate_hyde_document(user_query, domain)

    logger.info(f"원본: {user_query[:50]}...")
    logger.info(f"재작성: {rewritten}")
    logger.info(f"HyDE: {hyde_doc[:80]}...")

    return {
        "original_query": user_query,
        "rewritten_query": rewritten,
        "hyde_document": hyde_doc,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = reformulate("친구한테 300만원 빌려줬는데 1년째 안 갚고 연락도 안 돼요", "민사")
    print(f"\n원본: {result['original_query']}")
    print(f"재작성: {result['rewritten_query']}")
    print(f"HyDE: {result['hyde_document']}")
