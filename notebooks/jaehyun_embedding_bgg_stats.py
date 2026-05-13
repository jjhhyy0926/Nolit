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

INPUT_PATH  = DATA_DIR / "bgg_stats_final.csv"
INDEX_PATH  = DATA_DIR / "faiss_bgg_stats.index"
META_PATH   = DATA_DIR / "faiss_bgg_stats_meta.json"

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 100


# ── 1. CSV 로드 ───────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
texts = df["text"].tolist()
metas = df["metadata"].apply(json.loads).tolist()
print(f"[청크 로드] {len(texts):,}개")


# ── 2. 임베딩 생성 (배치 처리) ───────────────────────
def get_embeddings(texts: list, model: str, batch_size: int) -> np.ndarray:
    all_embeddings = []
    start = time.time()

    for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 중"):
        batch = texts[i:i + batch_size]

        while True:
            try:
                response = openai.embeddings.create(
                    input=batch,
                    model=model,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                time.sleep(0.5)  # 배치 사이 0.5초 대기
                break
            except openai.RateLimitError as e:
                wait = 5
                tqdm.write(f"Rate limit 도달 → {wait}초 대기 후 재시도...")
                time.sleep(wait)

    elapsed = time.time() - start
    print(f"[임베딩 소요시간] {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    return np.array(all_embeddings, dtype="float32")

embeddings = get_embeddings(texts, EMBED_MODEL, BATCH_SIZE)
print(f"[임베딩 완료] shape: {embeddings.shape}")


# ── 3. FAISS 인덱스 저장 ──────────────────────────────
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, str(INDEX_PATH))
print(f"[FAISS 저장] → {INDEX_PATH}")


# ── 4. metadata 저장 ──────────────────────────────────
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
print(f"[메타 저장] → {META_PATH}")

print(f"\n완료! 총 {len(texts):,}개 벡터 저장됨")