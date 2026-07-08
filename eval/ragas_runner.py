"""
RAGAS 자동 평가 하네스.
메트릭: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
결과 SQLite 저장 + JSON 리포트 출력.
"""
import json
import logging
import time
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from dotenv import load_dotenv

from graph.pipeline import run_pipeline
from logs.db import log_eval

load_dotenv()
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
TESTSET_PATH = EVAL_DIR / "testset.json"
REPORT_PATH = EVAL_DIR / "ragas_report.json"


def run_pipeline_for_eval(query: str) -> dict:
    """파이프라인 실행 후 RAGAS 포맷으로 변환."""
    result = run_pipeline(query)
    contexts = [s.get("summary", "") for s in result.get("summaries", [])]
    return {
        "question": query,
        "answer": result.get("final_answer", ""),
        "contexts": contexts if contexts else ["관련 판례를 찾을 수 없습니다."],
        "ground_truth": "",
        "log_id": result.get("log_id"),
    }


def load_testset() -> list[dict]:
    with open(TESTSET_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_eval(sample_size: int | None = None) -> dict:
    testset = load_testset()
    if sample_size:
        testset = testset[:sample_size]

    logger.info(f"RAGAS 평가 시작: {len(testset)}개 테스트 케이스")

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    log_ids = []

    for item in testset:
        logger.info(f"  [{item['id']}] {item['query'][:40]}...")
        try:
            pipeline_result = run_pipeline_for_eval(item["query"])
            eval_data["question"].append(pipeline_result["question"])
            eval_data["answer"].append(pipeline_result["answer"])
            eval_data["contexts"].append(pipeline_result["contexts"])
            eval_data["ground_truth"].append(item["ground_truth"])
            log_ids.append(pipeline_result["log_id"])
            time.sleep(1)  # rate limit
        except Exception as e:
            logger.error(f"케이스 {item['id']} 실패: {e}")
            eval_data["question"].append(item["query"])
            eval_data["answer"].append("")
            eval_data["contexts"].append([""])
            eval_data["ground_truth"].append(item["ground_truth"])
            log_ids.append(None)

    ds = Dataset.from_dict(eval_data)

    logger.info("RAGAS 점수 계산 중...")
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
        "sample_size": len(testset),
        "timestamp": time.time(),
    }

    # SQLite 저장
    for i, log_id in enumerate(log_ids):
        if log_id:
            try:
                log_eval(
                    query_id=log_id,
                    faithfulness=float(result["faithfulness"]),
                    answer_relevancy=float(result["answer_relevancy"]),
                    context_precision=float(result["context_precision"]),
                    context_recall=float(result["context_recall"]),
                )
            except Exception:
                pass

    # JSON 리포트
    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    return scores


def print_report(scores: dict) -> None:
    print("\n" + "=" * 50)
    print("RAGAS 평가 결과")
    print("=" * 50)
    print(f"  Faithfulness:       {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy:   {scores['answer_relevancy']:.4f}")
    print(f"  Context Precision:  {scores['context_precision']:.4f}")
    print(f"  Context Recall:     {scores['context_recall']:.4f}")
    print(f"  테스트 케이스:       {scores['sample_size']}개")
    print("=" * 50)

    targets = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
        "context_precision": 0.70,
        "context_recall": 0.65,
    }
    all_pass = True
    for metric, target in targets.items():
        status = "PASS" if scores[metric] >= target else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {metric}: {status} (목표={target}, 실제={scores[metric]:.4f})")

    print(f"\n최종: {'ALL PASS' if all_pass else 'FAIL — 개선 필요'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scores = run_eval(sample_size=5)  # 빠른 테스트는 5개만
    print_report(scores)
