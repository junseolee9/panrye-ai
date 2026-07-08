"""
통합 평가 러너 — RAGAS 4메트릭 + 결정적 루브릭.
저지 LLM: Groq llama-3.3-70b (무료) / --judge gemini 폴백.
임베딩: 로컬 ko-sroberta (answer_relevancy가 기본 OpenAI 임베딩을 쓰므로 반드시 오버라이드).

사용:
    python -m eval.runner --sample 3          # 소량 스모크
    python -m eval.runner                     # 전체 30케이스
    python -m eval.runner --judge gemini      # Gemini 저지
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from panrye.config import get_settings

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results.json"

# 성능 목표 (README 표와 동일)
TARGETS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.65,
}

_STATUTE_RE = re.compile(r"[가-힣]+법(?:률)?\s*제\d+조")
_CASE_NO_RE = re.compile(r"\d{2,4}[가-힣]{1,3}\d{2,7}")
_CITATION_RE = re.compile(r"\[판례\s*\d+\]")


def load_testset(sample: int | None = None) -> list[dict]:
    with open(EVAL_DIR / "testset.json", encoding="utf-8") as f:
        cases = json.load(f)
    if sample:
        # 도메인 고루 섞이도록 스트라이드 샘플링
        stride = max(1, len(cases) // sample)
        cases = cases[::stride][:sample]
    return cases


def rubric_checks(answer: str, contexts: list[str], expected_domain: str, domain: str) -> dict:
    """LLM 없이 판정 가능한 결정적 품질 체크."""
    context_text = " ".join(contexts)
    answer_case_nos = set(_CASE_NO_RE.findall(answer))
    hallucinated = sorted(n for n in answer_case_nos if n not in context_text)
    return {
        "domain_correct": domain == expected_domain,
        "statute_cited": bool(_STATUTE_RE.search(answer)),
        "precedent_cited": bool(_CITATION_RE.search(answer)),
        "disclaimer_present": "면책" in answer or "법률 자문이 아닙" in answer,
        "hallucinated_case_numbers": hallucinated,
    }


def run_cases(cases: list[dict]) -> list[dict]:
    """파이프라인 실행 + 루브릭. RAGAS 입력 rows 반환."""
    from panrye.graph.pipeline import run_pipeline

    rows = []
    for i, case in enumerate(cases, 1):
        logger.info(f"[{i}/{len(cases)}] {case['query'][:40]}")
        start = time.time()
        result = run_pipeline(case["query"])
        latency = time.time() - start

        contexts = [
            f"{s['case_name']} ({s['court']} {s['date']} {s.get('case_number', '')}) "
            f"{s['summary']}"
            for s in result["summaries"]
        ] or [""]

        rows.append({
            "case_id": case["id"],
            "expected_domain": case["domain"],
            "category": case["category"],
            "user_input": case["query"],
            "response": result["final_answer"],
            "retrieved_contexts": contexts,
            "reference": case["ground_truth"],
            "pipeline_domain": result["domain"],
            "pipeline_confidence": result["domain_confidence"],
            "latency_s": round(latency, 1),
            "log_id": result.get("log_id"),
            "rubric": rubric_checks(
                result["final_answer"], contexts, case["domain"], result["domain"]
            ),
        })
    return rows


def build_judge(judge: str):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()
    if judge == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model, google_api_key=settings.google_api_key, temperature=0
        )
    else:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=settings.llm_model, api_key=settings.groq_api_key, temperature=0
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def run_ragas(rows: list[dict], judge: str) -> dict:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm, emb = build_judge(judge)
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["user_input"],
            "response": r["response"],
            "retrieved_contexts": r["retrieved_contexts"],
            "reference": r["reference"],
        }
        for r in rows
    ])

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
        # Groq 무료 티어 rate limit 대응: 직렬 실행 + 넉넉한 타임아웃
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=8),
    )

    df = result.to_pandas()
    per_case = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
    return {
        "means": {k: round(float(per_case[k].mean()), 4) for k in per_case.columns},
        "per_case": [
            {k: (None if v != v else round(float(v), 4)) for k, v in row.items()}
            for row in per_case.to_dict(orient="records")
        ],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="판례.ai 통합 평가")
    parser.add_argument("--sample", type=int, default=None, help="케이스 수 제한")
    parser.add_argument("--judge", choices=["groq", "gemini"], default="groq")
    parser.add_argument("--skip-ragas", action="store_true", help="루브릭만 실행")
    args = parser.parse_args()

    cases = load_testset(args.sample)
    logger.info(f"평가 시작: {len(cases)}케이스, judge={args.judge}")

    rows = run_cases(cases)

    ragas_result = None
    if not args.skip_ragas:
        ragas_result = run_ragas(rows, args.judge)
        # per-case 점수를 rows에 합류 + DB 기록
        from panrye.storage.db import init_db, log_eval

        init_db()
        for row, scores in zip(rows, ragas_result["per_case"], strict=True):
            row["ragas"] = scores
            if row.get("log_id"):
                log_eval(
                    query_id=row["log_id"],
                    faithfulness=scores.get("faithfulness") or 0.0,
                    answer_relevancy=scores.get("answer_relevancy") or 0.0,
                    context_precision=scores.get("context_precision") or 0.0,
                    context_recall=scores.get("context_recall") or 0.0,
                    eval_model=f"ragas/{args.judge}",
                )

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge": args.judge,
        "n_cases": len(rows),
        "ragas_means": ragas_result["means"] if ragas_result else None,
        "targets": TARGETS,
        "rubric_summary": {
            "domain_accuracy": sum(r["rubric"]["domain_correct"] for r in rows) / len(rows),
            "statute_cited": sum(r["rubric"]["statute_cited"] for r in rows) / len(rows),
            "precedent_cited": sum(r["rubric"]["precedent_cited"] for r in rows) / len(rows),
            "disclaimer_present": sum(r["rubric"]["disclaimer_present"] for r in rows) / len(rows),
            "hallucinated_case_no_count": sum(
                len(r["rubric"]["hallucinated_case_numbers"]) for r in rows
            ),
        },
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 1),
        "cases": rows,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"결과 저장: {RESULTS_PATH}")

    print("\n=== 평가 요약 ===")
    if ragas_result:
        for k, v in ragas_result["means"].items():
            mark = "✅" if v >= TARGETS[k] else "❌"
            print(f"  {mark} {k}: {v:.3f} (목표 {TARGETS[k]})")
    for k, v in output["rubric_summary"].items():
        print(f"  · {k}: {v if isinstance(v, int) else f'{v:.0%}'}")
    print(f"  · 평균 레이턴시: {output['avg_latency_s']}s")


if __name__ == "__main__":
    main()
