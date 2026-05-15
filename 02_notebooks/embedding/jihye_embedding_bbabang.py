import json
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
import faiss

# ── 0. 모델 로드 ─────────────────────────────────────────────────
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# ── 1. 데이터 로드 ──────────────────────────────────────────────
df_stats = pd.read_csv("bbabang_stats_final.csv")
df_reviews = pd.read_csv("bbabang_reviews_final.csv", low_memory=False)

print(f"stats: {len(df_stats)}개 / reviews: {len(df_reviews)}개")


# ════════════════════════════════════════════════════════════════
# PART 1. stats FAISS
# ════════════════════════════════════════════════════════════════

# ── 2. 리뷰 집계 (title+store 기준) ─────────────────────────────
def summarize_reviews(group):
    valid_headcount = group[group["review_headcount"] > 0]["review_headcount"]
    avg_headcount = valid_headcount.mean() if len(valid_headcount) > 0 else None
    return pd.Series({
        "avg_headcount": round(avg_headcount, 2) if avg_headcount else None,
    })

review_agg = df_reviews.groupby(["title", "store_name"]).apply(summarize_reviews).reset_index()

# ── 3. fuzzy matching (title+store 기준) ─────────────────────────
review_key_strs = [f"{t}_{s}" for t, s in zip(review_agg["title"], review_agg["store_name"])]

def fuzzy_match_key(stats_title, stats_store, candidates, threshold=85):
    query = f"{stats_title}_{stats_store}"
    result = process.extractOne(query, candidates, scorer=fuzz.ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None

df_stats["match_key"] = df_stats.apply(
    lambda r: fuzzy_match_key(r["title"], r["store_name"], review_key_strs), axis=1
)

unmatched = df_stats[df_stats["match_key"].isna()][["title", "store_name"]]
print(f"리뷰 매칭 실패 {len(unmatched)}개:")
for _, r in unmatched.iterrows():
    print(f"  - {r['title']} / {r['store_name']}")

review_agg["match_key"] = [f"{t}_{s}" for t, s in zip(review_agg["title"], review_agg["store_name"])]
df = df_stats.merge(review_agg.drop(columns=["title", "store_name"]), on="match_key", how="left")

# ── 4. stats 임베딩 문서 생성 ────────────────────────────────────
def build_stats_document(row):
    parts = [f"테마명: {row['title']}"]

    if pd.notna(row.get("description")):
        desc = str(row["description"]).replace(row["title"], "").strip()
        if desc:
            parts.append(f"설명: {desc}")

    scores = []
    for col, label in [
        ("difficulty", "난이도"), ("horror", "공포도"), ("activity", "활동성"),
        ("satisfaction", "만족도"), ("puzzle", "퍼즐"), ("story", "스토리"),
    ]:
        if pd.notna(row.get(col)):
            scores.append(f"{label} {row[col]}")
    if scores:
        parts.append(", ".join(scores))

    if pd.notna(row.get("playing_time")):
        parts.append(f"플레이타임 {int(row['playing_time'])}분")

    if pd.notna(row.get("max_players")):
        parts.append(f"최대 인원 {int(row['max_players'])}명")

    if pd.notna(row.get("area")):
        parts.append(f"지역 {row['area']} {row['location']}")

    if pd.notna(row.get("avg_headcount")) and row["avg_headcount"] > 0:
        parts.append(f"평균 플레이 인원 {round(row['avg_headcount'], 1)}명")

    return " | ".join(parts)

df["document"] = df.apply(build_stats_document, axis=1)

# ── 5. stats 임베딩 + FAISS 저장 ────────────────────────────────
print("\nstats 임베딩 시작...")
stats_embeddings = model.encode(df["document"].tolist(), show_progress_bar=True, normalize_embeddings=True)

dim = stats_embeddings.shape[1]
stats_index = faiss.IndexFlatIP(dim)
stats_index.add(stats_embeddings.astype(np.float32))
faiss.write_index(stats_index, "bbabang_stats.index")
print(f"stats FAISS 저장 완료 - 벡터 수: {stats_index.ntotal}")

# ── 6. stats 메타데이터 저장 ─────────────────────────────────────
stats_metadata = []
for i, row in df.reset_index(drop=True).iterrows():
    stats_metadata.append({
        "id": i,
        "source": "bbabang",
        "title": row["title"],
        "store_name": row["store_name"],
        "area": row["area"],
        "location": row["location"],
        "playing_time": int(row["playing_time"]),
        "max_players": None if pd.isna(row.get("max_players")) else int(row["max_players"]),
        "price": None if pd.isna(row.get("price")) else int(row["price"]),
        "difficulty": None if pd.isna(row.get("difficulty")) else row["difficulty"],
        "horror": None if pd.isna(row.get("horror")) else row["horror"],
        "activity": None if pd.isna(row.get("activity")) else row["activity"],
        "satisfaction": None if pd.isna(row.get("satisfaction")) else row["satisfaction"],
        "puzzle": None if pd.isna(row.get("puzzle")) else row["puzzle"],
        "story": None if pd.isna(row.get("story")) else row["story"],
        "interior": None if pd.isna(row.get("interior")) else row["interior"],
        "production": None if pd.isna(row.get("production")) else row["production"],
        "avg_headcount": None if pd.isna(row.get("avg_headcount")) else row["avg_headcount"],
    })

with open("bbabang_stats_metadata.json", "w", encoding="utf-8") as f:
    json.dump(stats_metadata, f, ensure_ascii=False, indent=2)
print(f"stats 메타데이터 저장 완료 - 총 {len(stats_metadata)}개")


# ════════════════════════════════════════════════════════════════
# PART 2. 리뷰 FAISS
# ════════════════════════════════════════════════════════════════

# ── 7. 유효 리뷰 필터링 (30자 이상) ─────────────────────────────
df_valid = (
    df_reviews[df_reviews["review_text"].astype(str).str.len() >= 30]
    .copy()
    .reset_index(drop=True)
)
print(f"\n유효 리뷰 수: {len(df_valid)}개")

# ── 8. 청크 생성 (title+store 기준 10개씩) ───────────────────────
chunks = []
for (title, store), group in df_valid.groupby(["title", "store_name"]):
    reviews = group["review_text"].astype(str).tolist()
    for i in range(0, len(reviews), 10):
        chunk_texts = reviews[i:i+10]
        chunk_doc = f"테마명: {title} | 매장: {store} | 후기: {' '.join(chunk_texts)}"
        chunks.append({
            "title": title,
            "store_name": store,
            "source": "bbabang",
            "chunk_index": i // 10,
            "document": chunk_doc,
        })

print(f"총 리뷰 청크 수: {len(chunks)}")

# ── 9. 리뷰 임베딩 + FAISS 저장 ─────────────────────────────────
print("리뷰 임베딩 시작...")
chunk_docs = [c["document"] for c in chunks]
review_embeddings = model.encode(chunk_docs, show_progress_bar=True, normalize_embeddings=True, batch_size=64)

reviews_index = faiss.IndexFlatIP(dim)
reviews_index.add(review_embeddings.astype(np.float32))
faiss.write_index(reviews_index, "bbabang_reviews.index")
print(f"리뷰 FAISS 저장 완료 - 벡터 수: {reviews_index.ntotal}")

# ── 10. 리뷰 메타데이터 저장 ─────────────────────────────────────
reviews_metadata = []
for i, chunk in enumerate(chunks):
    reviews_metadata.append({
        "id": i,
        "source": chunk["source"],
        "title": chunk["title"],
        "store_name": chunk["store_name"],
        "chunk_index": chunk["chunk_index"],
        "document": chunk["document"],
    })

with open("bbabang_reviews_metadata.json", "w", encoding="utf-8") as f:
    json.dump(reviews_metadata, f, ensure_ascii=False, indent=2)
print(f"리뷰 메타데이터 저장 완료 - 총 {len(reviews_metadata)}개")