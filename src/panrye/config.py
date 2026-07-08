"""
전역 설정. 모든 API 키·모델 ID·경로·검색 파라미터의 단일 소스.
환경변수(.env)로 오버라이드 가능 — 필드명 대문자가 env 이름 (예: chroma_dir → CHROMA_DIR).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API 키 ---
    groq_api_key: str = ""
    google_api_key: str = ""
    hf_token: str = ""
    law_api_key: str = ""

    # --- 모델 ---
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    reranker_model: str = "Dongjin-kr/ko-reranker"
    summarizer_model: str = "digit82/kobart-summarization"
    llm_model: str = "llama-3.3-70b-versatile"
    # 재작성·분류 등 경량 태스크용 — 70b와 쿼터 분리 + 저지연
    fast_llm_model: str = "llama-3.1-8b-instant"
    # 평가 저지용 — 생성(llm_model)과 쿼터 분리를 위해 별도 모델
    judge_model: str = "openai/gpt-oss-120b"
    gemini_model: str = "gemini-2.5-flash"

    # --- 경로 (인덱스 산출물은 artifacts/ 아래) ---
    raw_data_dir: Path = PROJECT_ROOT / "raw_data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    chroma_dir: Path = PROJECT_ROOT / "artifacts" / "chroma_db"
    bm25_path: Path = PROJECT_ROOT / "artifacts" / "bm25_index.pkl"
    chunks_path: Path = PROJECT_ROOT / "artifacts" / "chunks.json"
    db_path: Path = PROJECT_ROOT / "logs" / "panrye.db"

    # --- 인덱스 아티팩트 원격 저장소 (HF Dataset) ---
    index_repo_id: str = ""

    # --- 검색 파라미터 ---
    collection_name: str = "panrye_precedents"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5
    rrf_k: int = 60

    # --- 분류기 ---
    classifier_confidence_threshold: float = 0.6

    # --- 요약 ---
    summarize_max_cases: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
