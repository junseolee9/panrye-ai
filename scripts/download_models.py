"""HF 모델 3종을 이미지 빌드 시점에 캐시로 내려받는다 (Dockerfile RUN 단계).
런타임 콜드스타트에서 모델 다운로드를 제거한다."""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)


def main() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import pipeline

    from panrye.config import get_settings

    settings = get_settings()
    logging.info(f"임베딩: {settings.embedding_model}")
    SentenceTransformer(settings.embedding_model)
    logging.info(f"리랭커: {settings.reranker_model}")
    CrossEncoder(settings.reranker_model, max_length=512)
    logging.info(f"요약: {settings.summarizer_model}")
    pipeline("summarization", model=settings.summarizer_model, device=-1)
    logging.info("모델 캐시 완료")


if __name__ == "__main__":
    main()
