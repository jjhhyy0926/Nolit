"""
recommender/rag/embeddings.py

[ 역할 ]
  - FAISS 인덱스 생성 및 저장 전담
  - OpenAI (1536차원) / HuggingFace (768차원) 두 엔진 지원
  - config.yaml에서 설정, loader.py에서 데이터 수신

[ 사용법 ]
  python manage.py embed_contents --all
  python manage.py embed_contents --source bgg_stats
  python manage.py embed_contents --source bbabang_stats
"""

import json
import os
import time
import numpy as np
import faiss
import openai
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from .config import get_config, get_data_dir, get_embedding_cfg, list_sources
from .loader import load_source


# ══════════════════════════════════════════════
# 0. 초기화
# ══════════════════════════════════════════════

_cfg     = get_config()
_emb_cfg = get_embedding_cfg()

load_dotenv(Path(__file__).resolve().parent.parent.parent / _cfg["paths"]["env_file"])
openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = _emb_cfg["openai_model"]
HF_MODEL     = _emb_cfg["hf_model"]
BATCH_SIZE   = _emb_cfg["batch_size"]
MAX_CHARS    = _emb_cfg["max_chars"]
SLEEP_SEC    = _emb_cfg["sleep_sec"]

# HuggingFace 모델은 필요할 때만 로드 (메모리 절약)
_hf_model = None

def _get_hf_model():
    global _hf_model
    if _hf_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  HuggingFace 모델 로드 중: {HF_MODEL}")
        _hf_model = SentenceTransformer(HF_MODEL)
    return _hf_model


# ══════════════════════════════════════════════
# 1. OpenAI 임베딩
# ══════════════════════════════════════════════

def _truncate(text: str) -> str:
    if not isinstance(text, str):
        return " "
    return text[:MAX_CHARS]


def _openai_batch(batch: list[str]) -> list[list[float]]:
    """단일 배치 OpenAI 임베딩 — Rate limit / BadRequest 재시도"""
    while True:
        try:
            response = openai.embeddings.create(input=batch, model=OPENAI_MODEL)
            time.sleep(SLEEP_SEC)
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            tqdm.write("Rate limit → 5초 대기...")
            time.sleep(5)
        except openai.BadRequestError as e:
            tqdm.write(f"BadRequest → 개별 처리: {e}")
            results = []
            for t in batch:
                try:
                    r = openai.embeddings.create(input=[t[:10000]], model=OPENAI_MODEL)
                    results.append(r.data[0].embedding)
                except Exception as e2:
                    tqdm.write(f"개별 실패 → 빈 벡터: {e2}")
                    results.append([0.0] * 1536)
            return results


def _embed_openai(texts: list[str], use_ckpt: bool, ckpt_dir: Path) -> np.ndarray:
    """OpenAI 전체 임베딩 — 체크포인트 유무에 따라 분기"""
    texts = [_truncate(t) for t in texts]

    # 체크포인트 없는 버전 (소용량)
    if not use_ckpt:
        all_emb = []
        total   = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        start   = time.time()
        for i in tqdm(range(total), desc="  OpenAI 임베딩"):
            batch = texts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            all_emb.extend(_openai_batch(batch))
        print(f"  소요: {(time.time()-start)/60:.1f}분")
        return np.array(all_emb, dtype="float32")

    # 체크포인트 있는 버전 (대용량)
    ckpt_dir.mkdir(exist_ok=True)
    saved       = sorted(ckpt_dir.glob("chunk_*.npy"))
    start_idx   = len(saved) * 1000
    chunk_emb   = []
    chunk_idx   = len(saved)
    total_batch = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    start       = time.time()

    print(f"  체크포인트 {len(saved)}개 확인 → {start_idx:,}번부터 재개")

    for i in tqdm(range(start_idx // BATCH_SIZE, total_batch), desc="  OpenAI 임베딩"):
        batch = texts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        chunk_emb.extend(_openai_batch(batch))

        if len(chunk_emb) >= 1000:
            f = ckpt_dir / f"chunk_{chunk_idx:04d}.npy"
            np.save(str(f), np.array(chunk_emb[:1000], dtype="float32"))
            tqdm.write(f"  청크 저장: {f.name}")
            chunk_emb = chunk_emb[1000:]
            chunk_idx += 1

    if chunk_emb:
        f = ckpt_dir / f"chunk_{chunk_idx:04d}.npy"
        np.save(str(f), np.array(chunk_emb, dtype="float32"))

    all_chunks = sorted(ckpt_dir.glob("chunk_*.npy"))
    embeddings = np.concatenate([np.load(str(c)) for c in all_chunks], axis=0)
    print(f"  소요: {(time.time()-start)/60:.1f}분")
    return embeddings


# ══════════════════════════════════════════════
# 2. HuggingFace 임베딩 (bbabang 전용)
# ══════════════════════════════════════════════

def _embed_hf(texts: list[str]) -> np.ndarray:
    """HuggingFace SentenceTransformer 임베딩"""
    model = _get_hf_model()
    print(f"  HuggingFace 임베딩 시작... ({len(texts):,}개)")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64,
    )
    return embeddings.astype(np.float32)


# ══════════════════════════════════════════════
# 3. FAISS 저장
# ══════════════════════════════════════════════

def _save_faiss(
    embeddings : np.ndarray,
    metas      : list[dict],
    index_path : Path,
    meta_path  : Path,
    index_type : str = "L2",
):
    """
    FAISS 인덱스 + 메타데이터 저장.
    - OpenAI → IndexFlatL2  (L2 거리)
    - HuggingFace → IndexFlatIP (내적, normalize 후 코사인 유사도)
    """
    dim = embeddings.shape[1]

    if index_type == "IP":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    print(f"  [FAISS] {index_path.name}  ({index.ntotal:,}개 벡터, dim={dim})")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"  [메타]  {meta_path.name}  ({len(metas):,}개)")


# ══════════════════════════════════════════════
# 4. 실행 함수 (외부 호출용)
# ══════════════════════════════════════════════

def run_embedding(source_name: str):
    """
    단일 데이터소스 임베딩 실행.
    management command / 외부 스크립트에서 호출.
    """
    cfg      = get_config()
    sources  = cfg["sources"]

    if source_name not in sources:
        raise ValueError(
            f"알 수 없는 소스: '{source_name}'\n"
            f"사용 가능: {list(sources.keys())}"
        )

    src_cfg   = sources[source_name]
    engine    = src_cfg.get("engine", "openai")
    use_ckpt  = src_cfg.get("use_ckpt", False)
    data_dir  = get_data_dir()

    print(f"\n{'='*52}")
    print(f"  [{source_name}]  engine={engine}  ckpt={use_ckpt}")
    print(f"{'='*52}")

    # 1. 데이터 로드
    texts, metas = load_source(source_name)
    print(f"  로드 완료: {len(texts):,}개")

    # 2. 임베딩 생성
    ckpt_dir = data_dir / f"ckpt_{source_name}"

    if engine == "hf":
        embeddings = _embed_hf(texts)
        index_type = "IP"   # HuggingFace → 내적 (코사인 유사도)
    else:
        embeddings = _embed_openai(texts, use_ckpt, ckpt_dir)
        index_type = "L2"   # OpenAI → L2 거리

    print(f"  임베딩 shape: {embeddings.shape}")

    # 3. FAISS + 메타 저장
    _save_faiss(
        embeddings,
        metas,
        index_path = data_dir / src_cfg["index"],
        meta_path  = data_dir / src_cfg["meta"],
        index_type = index_type,
    )

    print(f"\n  ✅ [{source_name}] 완료!\n")


def run_all():
    """config.yaml에 정의된 모든 데이터소스 순차 임베딩"""
    sources = list_sources()
    print(f"\n전체 임베딩 시작: {sources}\n")
    for source_name in sources:
        run_embedding(source_name)


# ══════════════════════════════════════════════
# 5. retriever.py에서 사용하는 로드 함수
# ══════════════════════════════════════════════

def load_index(source_name: str) -> tuple[faiss.Index, list]:
    """
    저장된 FAISS 인덱스 + 메타데이터 로드.
    retriever.py에서 검색 시 사용.

    Returns:
        (faiss.Index, list[dict])
    """
    cfg      = get_config()
    src_cfg  = cfg["sources"][source_name]
    data_dir = get_data_dir()

    index_path = data_dir / src_cfg["index"]
    meta_path  = data_dir / src_cfg["meta"]

    if not index_path.exists():
        raise FileNotFoundError(
            f"인덱스 없음: {index_path}\n"
            f"먼저 실행: python manage.py embed_contents --source {source_name}"
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metas = json.load(f)

    return index, metas


def get_query_embedding(text: str, engine: str = "openai") -> np.ndarray:
    """
    쿼리 텍스트 임베딩 생성.
    retriever.py에서 검색 쿼리 임베딩 시 사용.

    Args:
        text   : 검색 쿼리 텍스트
        engine : "openai" | "hf"

    Returns:
        shape (1, dim) float32 ndarray
    """
    if engine == "hf":
        model = _get_hf_model()
        vec   = model.encode([text], normalize_embeddings=True)
        return vec.astype(np.float32)
    else:
        response = openai.embeddings.create(input=[text], model=OPENAI_MODEL)
        vec      = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec
