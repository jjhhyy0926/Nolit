"""
recommender/rag/config.py

[ 역할 ]
  - config.yaml 로드 전담
  - 모든 모듈(loader, embeddings, retriever)에서 import해서 사용
  - lru_cache로 최초 1회만 파일 읽고 이후 캐싱
"""

import yaml
from pathlib import Path
from functools import lru_cache

# config.yaml 위치: Nolit 루트
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    """config.yaml 로드 (캐싱 적용)"""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml 없음: {CONFIG_PATH}\n"
            f"Nolit 루트에 config.yaml이 있어야 합니다."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    return get_base_dir() / get_config()["paths"]["data_dir"]


def get_embedding_cfg() -> dict:
    return get_config()["embedding"]


def get_retrieval_cfg() -> dict:
    return get_config()["retrieval"]


def list_sources() -> list[str]:
    return list(get_config()["sources"].keys())
