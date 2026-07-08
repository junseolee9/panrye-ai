"""
SQLite 쿼리 로깅.
저장: 쿼리/도메인/재작성/검색수/점수/레이턴시/답변.
RAGAS eval 결과도 저장.
"""
from __future__ import annotations
import json
import sqlite3
import time
import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# cwd 무관하게 이 파일 위치 기준으로 고정
DB_PATH = Path(__file__).resolve().parent / "panrye.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            user_query TEXT NOT NULL,
            domain TEXT,
            rewritten_query TEXT,
            retrieved_count INTEGER DEFAULT 0,
            summaries_json TEXT,
            final_answer TEXT,
            latency_ms REAL,
            feedback INTEGER  -- 1=helpful, 0=not helpful, NULL=no feedback
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            query_id INTEGER REFERENCES queries(id),
            faithfulness REAL,
            answer_relevancy REAL,
            context_precision REAL,
            context_recall REAL,
            eval_model TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
        CREATE INDEX IF NOT EXISTS idx_queries_domain ON queries(domain);
        """)
    logger.info(f"DB 초기화 완료: {DB_PATH}")


def log_query(
    user_query: str,
    domain: str,
    rewritten_query: str,
    retrieved_count: int,
    summaries: list[dict],
    final_answer: str,
    latency_ms: float | None = None,
) -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            """INSERT INTO queries
               (timestamp, user_query, domain, rewritten_query, retrieved_count,
                summaries_json, final_answer, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                user_query,
                domain,
                rewritten_query,
                retrieved_count,
                json.dumps(summaries, ensure_ascii=False),
                final_answer,
                latency_ms,
            ),
        )
        return cursor.lastrowid


def log_eval(
    query_id: int,
    faithfulness: float,
    answer_relevancy: float,
    context_precision: float,
    context_recall: float,
    eval_model: str = "ragas",
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO eval_results
               (timestamp, query_id, faithfulness, answer_relevancy,
                context_precision, context_recall, eval_model)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), query_id, faithfulness, answer_relevancy,
             context_precision, context_recall, eval_model),
        )


def update_feedback(query_id: int, helpful: bool) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE queries SET feedback = ? WHERE id = ?",
            (1 if helpful else 0, query_id),
        )


def get_recent_queries(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_eval_stats() -> dict:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_evals,
                AVG(faithfulness) as avg_faithfulness,
                AVG(answer_relevancy) as avg_relevancy,
                AVG(context_precision) as avg_precision,
                AVG(context_recall) as avg_recall
            FROM eval_results
        """).fetchone()
    return dict(row) if row else {}


# DB 자동 초기화
init_db()
