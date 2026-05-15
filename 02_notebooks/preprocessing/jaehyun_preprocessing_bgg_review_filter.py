import pandas as pd
import json
import csv
from pathlib import Path

INPUT_PATH  = Path(r"C:\lecture\Nolit\data\bgg_reviews_final.csv")
OUTPUT_PATH = Path(r"C:\lecture\Nolit\data\bgg_reviews_final1.csv")

# ── 1. 로드 ───────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", low_memory=False)
print(f"[원본] {len(df):,}개")

# ── 2. 임시 컬럼 추가 ─────────────────────────────────
df["title"] = df["metadata"].apply(lambda x: json.loads(x).get("title", "unknown"))
df["word_count"] = df["text"].fillna("").astype(str).apply(lambda x: len(x.split()))

# ── 3. 게임별 샘플링 ──────────────────────────────────
def sample_game(group, n=200):
    if len(group) <= n:
        return group  # 200개 이하 → 건드리지 않음

    filtered = group[group["word_count"] >= 100]  # 100단어 이상 필터

    if len(filtered) <= n:
        return filtered  # 필터 후 200개 이하면 그대로

    return filtered.nlargest(n, "word_count")  # 200개 초과면 긴 것 상위 200개

df_filtered = (
    df.groupby("title", group_keys=False)
      .apply(sample_game)
      .reset_index(drop=True)
)

print(f"[필터 후] {len(df_filtered):,}개")
print(f"[게임 수] {df_filtered['title'].nunique():,}개")

# ── 4. 임시 컬럼 제거 후 저장 ────────────────────────
df_filtered = df_filtered.drop(columns=["title", "word_count"])
df_filtered.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)
print(f"[저장 완료] → {OUTPUT_PATH}")