"""
eval/results.json → eval/report.md 생성 + README 메트릭 표 자동 주입.
README에는 <!-- METRICS:START --> ... <!-- METRICS:END --> 마커가 있어야 한다.

사용: python -m eval.report
"""
from __future__ import annotations

import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
RESULTS_PATH = EVAL_DIR / "results.json"
REPORT_PATH = EVAL_DIR / "report.md"
README_PATH = PROJECT_ROOT / "README.md"

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def metrics_table(data: dict) -> str:
    lines = ["| 메트릭 | 결과 | 목표 | 판정 |", "|---|---|---|---|"]
    means = data.get("ragas_means") or {}
    for key, label in METRIC_LABELS.items():
        v = means.get(key)
        target = data["targets"][key]
        if v is None:
            lines.append(f"| {label} | — | {target:.2f} | — |")
        else:
            mark = "✅" if v >= target else "❌"
            lines.append(f"| {label} | **{v:.3f}** | {target:.2f} | {mark} |")

    r = data["rubric_summary"]
    lines += [
        "",
        "| 루브릭 (결정적 체크) | 결과 |",
        "|---|---|",
        f"| 도메인 분류 정확도 | {r['domain_accuracy']:.0%} |",
        f"| 법조문 인용률 | {r['statute_cited']:.0%} |",
        f"| 판례 인용률 | {r['precedent_cited']:.0%} |",
        f"| 면책 고지 포함률 | {r['disclaimer_present']:.0%} |",
        f"| 허위 사건번호 | {r['hallucinated_case_no_count']}건 |",
        "",
        f"*{data['timestamp']} · {data['n_cases']}케이스 · judge={data['judge']}"
        f" · 평균 응답 {data['avg_latency_s']}s (CPU)*",
    ]
    return "\n".join(lines)


def per_case_table(data: dict) -> str:
    lines = [
        "| # | 도메인 | 분류 | Faith. | Relev. | Prec. | Recall | 응답(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in data["cases"]:
        ragas = c.get("ragas", {})

        def fmt(key: str, _r=ragas) -> str:
            v = _r.get(key)
            return f"{v:.2f}" if isinstance(v, float) else "—"

        ok = "✅" if c["rubric"]["domain_correct"] else "❌"
        lines.append(
            f"| {c['case_id']} | {c['expected_domain']} | {ok} {c['pipeline_domain']} "
            f"| {fmt('faithfulness')} | {fmt('answer_relevancy')} "
            f"| {fmt('context_precision')} | {fmt('context_recall')} | {c['latency_s']} |"
        )
    return "\n".join(lines)


def main() -> None:
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    table = metrics_table(data)

    report = (
        f"# 판례.ai 평가 리포트\n\n{table}\n\n## 케이스별 상세\n\n{per_case_table(data)}\n"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"리포트 저장: {REPORT_PATH}")

    if README_PATH.exists():
        readme = README_PATH.read_text(encoding="utf-8")
        pattern = re.compile(r"(<!-- METRICS:START -->).*?(<!-- METRICS:END -->)", re.S)
        if pattern.search(readme):
            readme = pattern.sub(rf"\1\n{table}\n\2", readme)
            README_PATH.write_text(readme, encoding="utf-8")
            print("README 메트릭 표 갱신 완료")
        else:
            print("README에 METRICS 마커 없음 — 건너뜀")


if __name__ == "__main__":
    main()
