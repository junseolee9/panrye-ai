"""
자체 LLM 저지 — 케이스당 1콜, 한국어 프롬프트, 강제 JSON.

RAGAS를 쓰지 않는 이유: RAGAS의 저지 프롬프트는 영어 기준이고 메트릭당 별도 LLM 콜을
발행한다. 무료 티어(비OpenAI) 저지에서는 (1) 한국어 답변에 대한 verdict 파싱이 자주
깨져 0/NaN이 나오고 (2) 케이스당 4콜이 rate limit을 태워 생성 쿼터까지 잠식했다.
단일 콜 + 한국어 루브릭 + JSON 강제 파싱으로 두 문제를 제거한다 (콜 수 1/4).

메트릭 정의(0.0~1.0, RAGAS 방법론 준용):
- faithfulness: 답변의 사실적 주장 중 제공된 판례 컨텍스트로 뒷받침되는 비율
- answer_relevancy: 답변이 질문의 법적 쟁점에 직접 응답하는 정도
- context_precision: 검색된 판례 중 이 질문 해결에 실제로 관련 있는 비율
- context_recall: 모범답안의 핵심 요소 중 컨텍스트가 커버하는 비율
"""
from __future__ import annotations

import json
import logging
import re
import time

from panrye.config import get_settings

logger = logging.getLogger(__name__)

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

JUDGE_PROMPT = """당신은 한국 법률 RAG 시스템의 품질 평가자입니다.
아래 자료를 읽고 4개 지표를 0.0~1.0 사이 소수로 채점하세요.

[사용자 질문]
{question}

[검색된 판례 컨텍스트]
{contexts}

[시스템 답변]
{answer}

[모범답안 (참고 기준)]
{reference}

채점 기준:
1. faithfulness — 답변의 사실적 주장(판례 내용, 법리) 중 위 컨텍스트로 뒷받침되는 비율.
   컨텍스트에 없는 판례 내용을 지어냈으면 크게 감점. 일반적 법률 상식(절차 안내 등)은 중립.
2. answer_relevancy — 답변이 질문의 법적 쟁점에 직접 응답하는가. 동문서답·불필요한 내용이면 감점.
3. context_precision — 검색된 판례들이 이 질문 해결에 관련 있는 비율. 무관한 판례가 섞이면 감점.
4. context_recall — 모범답안의 핵심 법리·조문 요소 중 컨텍스트가 담고 있는 비율.

반드시 아래 JSON만 출력 (설명 금지):
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, \
"context_recall": 0.0, "rationale": "한 문장 근거"}}"""


def _extract_json(text: str) -> dict:
    """응답에서 첫 JSON 오브젝트 추출 (코드펜스/사고과정 방어)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"JSON 없음: {text[:120]!r}")
    return json.loads(m.group(0))


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(
        settings.gemini_model,
        generation_config={"response_mime_type": "application/json", "temperature": 0.0},
    )
    return model.generate_content(prompt).text


def _call_groq(prompt: str) -> str:
    from groq import Groq

    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)
    r = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content


def judge_case(
    question: str,
    contexts: list[str],
    answer: str,
    reference: str,
    judge: str = "gemini",
    max_retries: int = 3,
) -> dict:
    """단일 케이스 채점. 반환: {faithfulness, answer_relevancy, context_precision,
    context_recall, rationale} — 실패 시 값들이 None."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        contexts="\n\n".join(contexts)[:6000],
        answer=answer[:3000],
        reference=reference,
    )
    call = _call_gemini if judge == "gemini" else _call_groq

    for attempt in range(1, max_retries + 1):
        try:
            raw = call(prompt)
            data = _extract_json(raw)
            scores = {}
            for k in METRIC_KEYS:
                v = float(data[k])
                if not 0.0 <= v <= 1.0:
                    raise ValueError(f"{k} 범위 밖: {v}")
                scores[k] = round(v, 4)
            scores["rationale"] = str(data.get("rationale", ""))[:300]
            return scores
        except Exception as e:
            wait = 10 * attempt
            logger.warning(
                f"저지 실패 (시도 {attempt}/{max_retries}): {str(e)[:120]} — {wait}s 대기"
            )
            time.sleep(wait)

    return {**dict.fromkeys(METRIC_KEYS), "rationale": "저지 실패"}


def judge_all(rows: list[dict], judge: str = "gemini", pause_s: float = 7.0) -> dict:
    """전체 케이스 채점. 무료 티어 RPM 대응으로 콜 간 간격 유지."""
    per_case = []
    for i, row in enumerate(rows, 1):
        logger.info(f"저지 [{i}/{len(rows)}] {row['user_input'][:30]}")
        scores = judge_case(
            row["user_input"], row["retrieved_contexts"], row["response"],
            row["reference"], judge=judge,
        )
        per_case.append(scores)
        if i < len(rows):
            time.sleep(pause_s)

    means = {}
    for k in METRIC_KEYS:
        vals = [c[k] for c in per_case if c[k] is not None]
        means[k] = round(sum(vals) / len(vals), 4) if vals else None
    return {"means": means, "per_case": per_case, "judged": judge}
