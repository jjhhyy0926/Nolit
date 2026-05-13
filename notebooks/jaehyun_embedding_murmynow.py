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
BASE_DIR    = Path("C:/lecture/NOLIT")
DATA_DIR    = BASE_DIR / "data"
ENV_PATH    = BASE_DIR / ".env"

load_dotenv(ENV_PATH)
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_PATH   = DATA_DIR / "murmynow_final.csv"
INDEX_PATH   = DATA_DIR / "faiss_murmynow.index"
META_PATH    = DATA_DIR / "faiss_murmynow_meta.json"
CKPT_PATH    = DATA_DIR / "faiss_murmynow_ckpt.npy"

EMBED_MODEL  = "text-embedding-3-small"
BATCH_SIZE   = 100


# ── 1. CSV 로드 ───────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", low_memory=False)
texts = df["text"].fillna(" ").astype(str).tolist()
metas = df["metadata"].apply(json.loads).tolist()
print(f"[청크 로드] {len(texts):,}개")


# ── 2. 체크포인트 로드 (이어하기) ────────────────────
start_batch    = 0
all_embeddings = []

if CKPT_PATH.exists():
    all_embeddings = np.load(str(CKPT_PATH), allow_pickle=True).tolist()
    start_batch = len(all_embeddings) // BATCH_SIZE
    print(f"[체크포인트 복원] {len(all_embeddings):,}개 → {start_batch}번 배치부터 재개")
else:
    print("[체크포인트 없음] 처음부터 시작")


# ── 3. 임베딩 생성 ────────────────────────────────────
start = time.time()
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(start_batch, total_batches), desc="임베딩 중", initial=start_batch, total=total_batches):
    batch = texts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    batch = [t[:30000] if len(t) > 30000 else t for t in batch]  # 토큰 초과 방지

    while True:
        try:
            response = openai.embeddings.create(
                input=batch,
                model=EMBED_MODEL,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.5)
            break
        except openai.RateLimitError:
            tqdm.write("Rate limit 도달 → 5초 대기 후 재시도...")
            time.sleep(5)
        except openai.BadRequestError as e:
            tqdm.write(f"BadRequest → 개별 처리로 전환: {e}")
            for t in batch:
                try:
                    r = openai.embeddings.create(input=[t[:10000]], model=EMBED_MODEL)
                    all_embeddings.append(r.data[0].embedding)
                except Exception as e2:
                    tqdm.write(f"개별 처리 실패 → 빈 벡터 삽입: {e2}")
                    all_embeddings.append([0.0] * 1536)
            break

    # 100배치마다 체크포인트 저장
    if (i + 1) % 100 == 0:
        np.save(str(CKPT_PATH), np.array(all_embeddings, dtype="float32"))
        tqdm.write(f"[체크포인트 저장] {len(all_embeddings):,}개")

elapsed = time.time() - start
print(f"[임베딩 소요시간] {elapsed:.1f}초 ({elapsed/60:.1f}분)")


# ── 4. FAISS 인덱스 저장 ──────────────────────────────
embeddings = np.array(all_embeddings, dtype="float32")
print(f"[임베딩 완료] shape: {embeddings.shape}")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, str(INDEX_PATH))
print(f"[FAISS 저장] → {INDEX_PATH}")


# ── 5. metadata 저장 ──────────────────────────────────
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
print(f"[메타 저장] → {META_PATH}")

# 체크포인트 삭제
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
    print("[체크포인트 삭제 완료]")

print(f"\n완료! 총 {len(texts):,}개 벡터 저장됨")