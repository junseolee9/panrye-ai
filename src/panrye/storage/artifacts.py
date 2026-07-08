"""
인덱스 아티팩트 원격 저장 (HF Dataset repo).
- upload_artifacts(): chroma tar + bm25 pickle + chunks.json + MANIFEST(sha256) 업로드
- download_artifacts(): snapshot_download → untar → 해시 검증
빌드/배포 커플링 없이 인덱스만 교체 가능하게 한다 (Spaces 콜드스타트 재임베딩 회피).
"""
from __future__ import annotations

import hashlib
import json
import logging
import tarfile
import tempfile
from pathlib import Path

from panrye.config import get_settings

logger = logging.getLogger(__name__)

CHROMA_TAR = "chroma_db.tar.gz"
BM25_FILE = "bm25_index.pkl"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "MANIFEST.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def upload_artifacts(repo_id: str | None = None) -> None:
    from huggingface_hub import HfApi

    settings = get_settings()
    repo_id = repo_id or settings.index_repo_id
    if not repo_id:
        raise ValueError("INDEX_REPO_ID 미설정")

    api = HfApi(token=settings.hf_token or None)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / CHROMA_TAR
        logger.info(f"chroma_db 압축 중: {settings.chroma_dir}")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(settings.chroma_dir, arcname="chroma_db")

        files = {
            CHROMA_TAR: tar_path,
            BM25_FILE: settings.bm25_path,
            CHUNKS_FILE: settings.chunks_path,
        }
        manifest = {
            "files": {name: _sha256(p) for name, p in files.items()},
            "chunk_count": len(json.loads(settings.chunks_path.read_text(encoding="utf-8"))),
        }
        manifest_path = Path(tmp) / MANIFEST_FILE
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        for name, path in {**files, MANIFEST_FILE: manifest_path}.items():
            logger.info(f"업로드: {name} ({path.stat().st_size / 1e6:.1f}MB)")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
            )
    logger.info(f"업로드 완료: https://huggingface.co/datasets/{repo_id}")


def download_artifacts(repo_id: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    settings = get_settings()
    repo_id = repo_id or settings.index_repo_id
    if not repo_id:
        raise ValueError("INDEX_REPO_ID 미설정")

    local = Path(
        snapshot_download(repo_id=repo_id, repo_type="dataset", token=settings.hf_token or None)
    )
    manifest = json.loads((local / MANIFEST_FILE).read_text(encoding="utf-8"))

    for name in (CHROMA_TAR, BM25_FILE, CHUNKS_FILE):
        actual = _sha256(local / name)
        expected = manifest["files"][name]
        if actual != expected:
            raise RuntimeError(f"아티팩트 해시 불일치: {name}")

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger.info("chroma_db 압축 해제 중...")
    with tarfile.open(local / CHROMA_TAR, "r:gz") as tar:
        tar.extractall(settings.artifacts_dir, filter="data")

    for name, dest in ((BM25_FILE, settings.bm25_path), (CHUNKS_FILE, settings.chunks_path)):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((local / name).read_bytes())

    logger.info(f"아티팩트 준비 완료: {settings.artifacts_dir} (청크 {manifest['chunk_count']}개)")
