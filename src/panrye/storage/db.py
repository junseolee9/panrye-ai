"""
SQLite 쿼리 로깅.
저장: 쿼리/도메인/재작성/검색수/점수/레이턴시/답변.
RAGAS eval 결과도 저장.
init_db()는 앱 startup 또는 CLI에서 명시적으로 호출한다 (import 부작용 없음).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager

from panrye.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def get_conn():
    db_path = get_settings().db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
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
    logger.info(f"DB 초기화 완료: {get_settings().db_path}")


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


def get_query_stats() -> dict:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_queries,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) as helpful_count,
                SUM(CASE WHEN feedback = 0 THEN 1 ELSE 0 END) as unhelpful_count
            FROM queries
        """).fetchone()
    return dict(row) if row else {}
