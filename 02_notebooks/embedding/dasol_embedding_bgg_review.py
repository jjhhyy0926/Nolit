import json
import numpy as np
import faiss
import openai
import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm


# ── 설정 ─────────────────────────────────────────────
BASE_DIR = Path(r"D:\project\Nolit\notebooks")
DATA_DIR    = BASE_DIR / "data"
ENV_PATH    = BASE_DIR / ".env"

load_dotenv(ENV_PATH)
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_PATH   = DATA_DIR / "bgg_reviews_final.csv"
INDEX_PATH   = DATA_DIR / "faiss_bgg_reviews.index"
META_PATH    = DATA_DIR / "faiss_bgg_reviews_meta.json"
CKPT_PATH    = DATA_DIR / "faiss_bgg_reviews_ckpt.npy"  # 체크포인트

EMBED_MODEL  = "text-embedding-3-small"
BATCH_SIZE   = 100
MAX_TOKENS   = 8000  # 8192 한도보다 여유있게


# ── 1. CSV 로드 ───────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", low_memory=False)
texts = df["text"].tolist()
metas = df["metadata"].apply(json.loads).tolist()
print(f"[청크 로드] {len(texts):,}개")


# ── 2. 텍스트 토큰 초과 방지용 자르기 ────────────────
def truncate_text(text: str, max_chars: int = 30000) -> str:
    """토큰 계산 없이 문자 수 기준으로 자름 (1토큰 ≈ 4자)"""
    if not isinstance(text, str):
        return " "
    return text[:max_chars] if len(text) > max_chars else text

texts = [truncate_text(t) for t in texts]


# ── 3. 체크포인트 로드 (이어하기) ────────────────────
CKPT_DIR = DATA_DIR / "ckpt_chunks"
CKPT_DIR.mkdir(exist_ok=True)

# 저장된 청크 파일 목록 확인
saved_chunks = sorted(CKPT_DIR.glob("chunk_*.npy"))
start_idx = len(saved_chunks) * 1000  # 청크당 1000개
print(f"[체크포인트 복원] 청크 {len(saved_chunks)}개 → {start_idx:,}번부터 재개")


# ── 4. 임베딩 생성 ────────────────────────────────────
start = time.time()
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE  # ← 이거 추가
chunk_embeddings = []  # 현재 청크 임시 저장
chunk_idx = len(saved_chunks)

for i in tqdm(range(start_idx // BATCH_SIZE, total_batches), desc="임베딩 중"):
    batch = texts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

    while True:
        try:
            response = openai.embeddings.create(input=batch, model=EMBED_MODEL)
            chunk_embeddings.extend([item.embedding for item in response.data])
            time.sleep(0.5)
            break
        except openai.RateLimitError:
            tqdm.write("Rate limit → 5초 대기...")
            time.sleep(5)
        except openai.BadRequestError as e:
            for t in batch:
                try:
                    r = openai.embeddings.create(input=[t[:10000]], model=EMBED_MODEL)
                    chunk_embeddings.append(r.data[0].embedding)
                except:
                    chunk_embeddings.append([0.0] * 1536)
            break

    # 1000개(=10배치)마다 청크 파일로 저장하고 메모리 비우기
    if len(chunk_embeddings) >= 1000:
        chunk_path = CKPT_DIR / f"chunk_{chunk_idx:04d}.npy"
        np.save(str(chunk_path), np.array(chunk_embeddings[:1000], dtype="float32"))
        tqdm.write(f"[청크 저장] {chunk_path.name} ({(chunk_idx+1)*1000:,}개 완료)")
        chunk_embeddings = chunk_embeddings[1000:]  # 저장한 것 제거
        chunk_idx += 1


# ── 5. 모든 청크 합쳐서 FAISS 저장 ──────────────────
tqdm.write("청크 합치는 중...")
all_chunks = sorted(CKPT_DIR.glob("chunk_*.npy"))
embeddings = np.concatenate([np.load(str(c)) for c in all_chunks], axis=0)
print(f"[임베딩 완료] shape: {embeddings.shape}")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, str(INDEX_PATH))
print(f"[FAISS 저장] → {INDEX_PATH}")


# ── 6. metadata 저장 ──────────────────────────────────
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
print(f"[메타 저장] → {META_PATH}")

# 체크포인트 삭제
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
    print("[체크포인트 삭제 완료]")

print(f"\n완료! 총 {len(texts):,}개 벡터 저장됨")