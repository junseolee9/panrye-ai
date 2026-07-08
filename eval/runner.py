"""
통합 평가 러너 — LLM 저지 4메트릭 (RAGAS 방법론 준용, eval/judge.py) + 결정적 루브릭.
저지 LLM: Gemini 2.5-flash 기본 (생성용 Groq와 쿼터 분리) / --judge groq 전환.

사용:
    python -m eval.runner --sample 3          # 소량 스모크
    python -m eval.runner                     # 전체 30케이스
    python -m eval.runner --judge groq        # Groq 저지
    python -m eval.runner --skip-judge        # 루브릭만
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

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


_FALLBACK_MARKER = "일시적으로 사용할 수 없습니다"


def _run_with_retry(query: str, max_retries: int = 2) -> tuple[dict, float]:
    """LLM 쿼터 소진으로 폴백 답변이 나오면 대기 후 재시도.
    폴백 문구를 채점하면 지표 전체가 무효가 되므로 eval에서는 필수 방어."""
    from panrye.graph.pipeline import run_pipeline

    for attempt in range(max_retries + 1):
        start = time.time()
        result = run_pipeline(query)
        latency = time.time() - start
        if _FALLBACK_MARKER not in result["final_answer"]:
            return result, latency
        if attempt < max_retries:
            wait = 90 * (attempt + 1)
            logger.warning(f"생성 폴백 감지 (쿼터 추정) — {wait}s 대기 후 재시도")
            time.sleep(wait)
    return result, latency


CHECKPOINT_PATH = EVAL_DIR / "results-partial.json"


def _load_checkpoint() -> list[dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_checkpoint(rows: list[dict]) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def run_cases(cases: list[dict], pause_s: float = 10.0, resume: bool = False) -> list[dict]:
    """파이프라인 실행 + 루브릭. 케이스마다 체크포인트 저장 — 쿼터/중단 대비."""
    rows = _load_checkpoint() if resume else []
    done_ids = {r["case_id"] for r in rows}
    if done_ids:
        logger.info(f"체크포인트 재개: {len(done_ids)}케이스 완료됨")

    for i, case in enumerate(cases, 1):
        if case["id"] in done_ids:
            continue
        logger.info(f"[{i}/{len(cases)}] {case['query'][:40]}")
        result, latency = _run_with_retry(case["query"])
        if i < len(cases):
            time.sleep(pause_s)  # 무료 티어 분당 쿼터 존중

        # 생성기가 실제로 본 컨텍스트([판례 N] 블록)를 그대로 평가에 사용 —
        # 요약만 넘기면 faithfulness가 실제보다 낮게 측정됨
        contexts = [b for b in result["context"].split("\n\n") if b.strip()] or [""]

        # 고정 면책 보일러플레이트는 relevancy 측정을 희석하므로 제거
        from panrye.agents.generator import DISCLAIMER

        answer = result["final_answer"].replace(DISCLAIMER, "").strip()

        rows.append({
            "case_id": case["id"],
            "expected_domain": case["domain"],
            "category": case["category"],
            "user_input": case["query"],
            "response": answer,
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
        _save_checkpoint(rows)

    rows.sort(key=lambda r: r["case_id"])
    return rows


def run_judge(rows: list[dict], judge: str) -> dict:
    from eval.judge import judge_all

    return judge_all(rows, judge=judge)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="판례.ai 통합 평가")
    parser.add_argument("--sample", type=int, default=None, help="케이스 수 제한")
    parser.add_argument("--judge", choices=["groq", "gemini"], default="gemini")
    parser.add_argument("--skip-judge", action="store_true", help="루브릭만 실행")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 이어서")
    args = parser.parse_args()

    cases = load_testset(args.sample)
    logger.info(f"평가 시작: {len(cases)}케이스, judge={args.judge}")

    rows = run_cases(cases, resume=args.resume)

    judge_result = None
    if not args.skip_judge:
        judge_result = run_judge(rows, args.judge)
        # per-case 점수를 rows에 합류 + DB 기록
        from panrye.storage.db import init_db, log_eval

        init_db()
        for row, scores in zip(rows, judge_result["per_case"], strict=True):
            row["judge_scores"] = scores
            if row.get("log_id"):
                log_eval(
                    query_id=row["log_id"],
                    faithfulness=scores.get("faithfulness") or 0.0,
                    answer_relevancy=scores.get("answer_relevancy") or 0.0,
                    context_precision=scores.get("context_precision") or 0.0,
                    context_recall=scores.get("context_recall") or 0.0,
                    eval_model=f"judge/{args.judge}",
                )

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge": args.judge,
        "n_cases": len(rows),
        "judge_means": judge_result["means"] if judge_result else None,
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
    if judge_result:
        for k, v in judge_result["means"].items():
            if v is None:
                print(f"  — {k}: 저지 실패")
                continue
            mark = "✅" if v >= TARGETS[k] else "❌"
            print(f"  {mark} {k}: {v:.3f} (목표 {TARGETS[k]})")
    for k, v in output["rubric_summary"].items():
        print(f"  · {k}: {v if isinstance(v, int) else f'{v:.0%}'}")
    print(f"  · 평균 레이턴시: {output['avg_latency_s']}s")


if __name__ == "__main__":
    main()
