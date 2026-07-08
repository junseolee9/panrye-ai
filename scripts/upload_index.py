"""빌드된 인덱스 아티팩트를 HF Dataset repo로 업로드.

사용: INDEX_REPO_ID=<user>/panrye-index python scripts/upload_index.py
"""
from __future__ import annotations

import logging

from panrye.storage.artifacts import upload_artifacts

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    upload_artifacts()
