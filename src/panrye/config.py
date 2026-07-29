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

    # --- 추론 디바이스 ("" = 자동 감지, 또는 cpu/mps/cuda 강제) ---
    device: str = ""

    # --- API 보호 (공개 배포 대비) ---
    # SPA는 API와 같은 오리진에서 서빙되므로 기본은 교차 출처 차단. 쉼표로 구분해 허용.
    cors_origins: str = ""
    # 미설정이면 /api/stats 비활성 (운영 통계에 타인의 질문이 섞여 있다)
    stats_token: str = ""
    # IP당 분당 요청 수. 0이면 해제. 무료 LLM 쿼터를 외부인이 태우는 것을 막는다.
    rate_limit_per_min: int = 30

    # --- 분류기 ---
    classifier_confidence_threshold: float = 0.6

    # --- 요약 ---
    summarize_max_cases: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_device() -> str:
    """로컬 추론 디바이스. Mac은 mps, GPU 서버는 cuda, 배포(Spaces/Docker)는 cpu로 떨어진다."""
    if get_settings().device:
        return get_settings().device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_groq_client():
    """Groq 클라이언트 단일 인스턴스 — 호출마다 새로 만들면 TLS 핸드셰이크가 반복된다."""
    from groq import Groq

    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY 미설정. .env 확인하세요.")
    return Groq(api_key=settings.groq_api_key)
