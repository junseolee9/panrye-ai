"""
Agent 5: Answer Generator
판례 컨텍스트 기반 최종 답변 생성.
LLM: Groq (primary) → Gemini 1.5 Flash (fallback).
스트리밍 지원.
"""
from __future__ import annotations
import logging
import os
from typing import Generator

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

DISCLAIMER = """
---
⚠️ **법적 면책 고지**: 이 답변은 AI가 공개 판례를 기반으로 생성한 참고 정보이며, 정식 법률 자문이 아닙니다. 실제 법적 판단은 변호사나 법률 전문가와 상담하시기 바랍니다.
"""

SYSTEM_PROMPT = """당신은 한국 법률 전문 AI 어시스턴트입니다.
사용자의 상황과 관련 판례를 분석하여 실용적인 법률 정보를 제공합니다.

답변 형식 (반드시 준수):
1. **상황 분석**: 법적 쟁점을 구체적 법률 용어로 요약
2. **관련 판례**: [판례 N] 형식으로 핵심 내용 인용
3. **적용 법조문**: 반드시 관련 법조문을 명시 (예: 민법 제390조, 근로기준법 제23조)
4. **법적 판단 방향**: 판례와 법조문 기반 예상 결과
5. **권장 행동**: 실질적 단계 (신고처/절차 포함)

엄수 규칙:
- **법조문은 반드시 구체적으로 명시**: "민법 제OOO조", "근로기준법 제OO조" 형식
- 판례 인용 시 [판례 N] 출처 표시 필수
- 적용 법조문 없이 답변 금지
- 800자 이내
- 법률 용어는 괄호로 쉬운 설명 병기 (예: 채무불이행(계약 위반))
- 마크다운 제목(#, ##, ###)은 사용 금지. 굵은 글씨(**)와 목록만 사용
- 한국어로만 답변"""


def _generate_with_groq(user_query: str, context: str, stream: bool = False):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"[관련 판례]\n{context}\n\n[사용자 상황]\n{user_query}",
        },
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=1500,
        temperature=0.2,
        stream=stream,
    )

    if stream:
        return response
    return response.choices[0].message.content


def _generate_with_gemini(user_query: str, context: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"{SYSTEM_PROMPT}\n\n[관련 판례]\n{context}\n\n[사용자 상황]\n{user_query}"
    response = model.generate_content(prompt)
    return response.text


def generate_answer(user_query: str, context: str) -> str:
    """동기 답변 생성 (Groq → Gemini fallback)."""
    if GROQ_API_KEY:
        try:
            answer = _generate_with_groq(user_query, context, stream=False)
            return answer + DISCLAIMER
        except Exception as e:
            logger.warning(f"Groq 실패: {e}. Gemini로 전환.")

    if GOOGLE_API_KEY:
        try:
            answer = _generate_with_gemini(user_query, context)
            return answer + DISCLAIMER
        except Exception as e:
            logger.error(f"Gemini도 실패: {e}")

    return "현재 AI 서비스를 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해주세요." + DISCLAIMER


def stream_answer(user_query: str, context: str) -> Generator[str, None, None]:
    """스트리밍 답변 생성."""
    if not GROQ_API_KEY:
        yield generate_answer(user_query, context)
        return

    emitted = False
    try:
        stream = _generate_with_groq(user_query, context, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                emitted = True
                yield delta
        yield DISCLAIMER
    except Exception as e:
        logger.warning(f"스트리밍 실패: {e}. 일반 생성으로 대체.")
        if emitted:
            # 이미 부분 출력됨 — 전체 답변을 다시 붙이면 중복되므로 중단 안내만
            yield "\n\n(연결이 중단되어 답변이 불완전할 수 있습니다.)" + DISCLAIMER
        else:
            yield generate_answer(user_query, context)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dummy_context = """[판례 1]
사건명: 손해배상(기)
법원: 대법원 | 날짜: 2023.05.15 | 영역: 민사
참조법조: 민법 제390조, 민법 제393조
요약: 피고가 차용금 3,000,000원을 변제 기일 이후에도 상환하지 않아 법원이 채무불이행으로 인한 손해배상을 인용하였다."""

    print("=== 동기 생성 ===")
    answer = generate_answer("친구한테 300만원 빌려줬는데 안 갚아요", dummy_context)
    print(answer)
