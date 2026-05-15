import pandas as pd
import re
import csv
import json
from pathlib import Path


# ── 설정 ─────────────────────────────────────────────
DATA_DIR    = Path("C:/lecture/NOLIT/data")
INPUT_FILE  = DATA_DIR / "bgg_reviews_merged.csv"
CSV_OUTPUT  = DATA_DIR / "bgg_reviews_final.csv"
JSON_OUTPUT = DATA_DIR / "bgg_reviews_final.json"


# ── 1. 로드 ──────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)
print(f"[원본] {len(df):,}행")


# ── 2. 3단어 이하 리뷰 제거 ──────────────────────────
def count_words(text) -> int:
    """한글 어절 + 영문 단어 합산"""
    if not isinstance(text, str):
        return 0
    korean_words  = re.findall(r"[가-힣]+", text)
    english_words = re.findall(r"\b[a-zA-Z]+\b", text)
    return len(korean_words) + len(english_words)

mask = df["review"].apply(count_words) > 3
removed = (~mask).sum()
df = df[mask].reset_index(drop=True)

print(f"[제거됨] {removed:,}행")
print(f"[필터 후] {len(df):,}행")


# ── 3. chunk 생성 (text + metadata) ──────────────────
chunks = []

for _, row in df.iterrows():
    review = row["review"]
    if not isinstance(review, str) or str(review) == "nan":
        review = " "
    review = review.strip() or " "

    chunks.append({
        "text": review,   # 임베딩 대상
        "metadata": {
            "source":  "bgg",
            "type":    "review",
            "rank":    int(row["rank"]) if pd.notna(row.get("rank")) else None,
            "title":   row.get("title"),
            "rating":  float(row["rating"]) if pd.notna(row.get("rating")) else None,
        }
    })

print(f"[청크 생성] {len(chunks):,}개")


# ── 4. 저장 ──────────────────────────────────────────
# JSON
with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print(f"JSON 저장 완료 → {JSON_OUTPUT}")

# CSV (metadata는 JSON 문자열로 변환)
csv_rows = [
    {"text": c["text"], "metadata": json.dumps(c["metadata"], ensure_ascii=False)}
    for c in chunks
]
pd.DataFrame(csv_rows).to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)
print(f"CSV 저장 완료 → {CSV_OUTPUT}")