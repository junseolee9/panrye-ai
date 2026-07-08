# 판례.ai — HF Spaces (Docker SDK) / 로컬 공용 이미지
FROM python:3.11-slim

# kiwipiepy 빌드 등 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces 요구: non-root 사용자, 쓰기 가능한 HOME
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

COPY --chown=user requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user static ./static
COPY --chown=user scripts ./scripts
RUN pip install --no-cache-dir --user -e . --no-deps

# 모델 3종을 이미지에 베이크 (콜드스타트에서 다운로드 제거)
RUN python scripts/download_models.py

# 인덱스 아티팩트는 시작 시 INDEX_REPO_ID 데이터셋에서 부트스트랩됨
# (upload: scripts/upload_index.py). DB는 컨테이너 로컬 경로.
ENV DB_PATH=/home/user/app/panrye.db

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "panrye.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
