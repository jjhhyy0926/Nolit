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
BASE_DIR     = Path(r"C:\lecture\Nolit")
DATA_DIR     = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0
)

INPUT_PATH   = DATA_DIR / "bgg_reviews_final2.csv"
INDEX_PATH   = DATA_DIR / "faiss_bgg_reviews.index"
META_PATH    = DATA_DIR / "faiss_bgg_reviews_meta.json"

EMBED_MODEL  = "text-embedding-3-small"
BATCH_SIZE   = 100


# ── 1. CSV 로드 ───────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", low_memory=False)
texts = df["text"].fillna(" ").astype(str).tolist()
metas = df["metadata"].apply(json.loads).tolist()
print(f"[청크 로드] {len(texts):,}개")


# ── 2. 텍스트 자르기 ──────────────────────────────────
def truncate_text(text: str, max_chars: int = 30000) -> str:
    if not isinstance(text, str):
        return " "
    return text[:max_chars] if len(text) > max_chars else text

texts = [truncate_text(t) for t in texts]


# ── 3. FAISS 인덱스 초기화 ────────────────────────────
dimension = 1536
index = faiss.IndexFlatL2(dimension)
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE


# ── 4. 임베딩 생성 & 바로 FAISS에 추가 ───────────────
start = time.time()

for i in tqdm(range(total_batches), desc="임베딩 중"):
    batch = texts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

    while True:
        try:
            response = client.embeddings.create(
                input=batch,
                model=EMBED_MODEL,
            )
            batch_embeddings = np.array(
                [item.embedding for item in response.data], dtype="float32"
            )
            index.add(batch_embeddings)
            time.sleep(0.5)
            break
        except openai.RateLimitError:
            tqdm.write("Rate limit 도달 → 5초 대기 후 재시도...")
            time.sleep(5)
        except openai.APITimeoutError:
            tqdm.write("타임아웃 → 5초 대기 후 재시도...")
            time.sleep(5)
        except openai.BadRequestError as e:
            tqdm.write(f"BadRequest → 개별 처리로 전환: {e}")
            for t in batch:
                try:
                    r = client.embeddings.create(input=[t[:10000]], model=EMBED_MODEL)
                    vec = np.array([r.data[0].embedding], dtype="float32")
                    index.add(vec)
                except Exception as e2:
                    tqdm.write(f"개별 처리 실패 → 빈 벡터 삽입: {e2}")
                    index.add(np.zeros((1, 1536), dtype="float32"))
            break

elapsed = time.time() - start
print(f"[임베딩 소요시간] {elapsed:.1f}초 ({elapsed/60:.1f}분)")


# ── 5. FAISS 최종 저장 ────────────────────────────────
faiss.write_index(index, str(INDEX_PATH))
print(f"[FAISS 저장] → {INDEX_PATH}  (총 {index.ntotal:,}개)")


# ── 6. metadata 저장 ──────────────────────────────────
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
print(f"[메타 저장] → {META_PATH}")

print(f"\n완료! 총 {len(texts):,}개 벡터 저장됨")