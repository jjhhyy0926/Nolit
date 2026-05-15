import pandas as pd
import re
import csv
import json
from pathlib import Path


# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
DATA_DIR = Path("C:/lecture/NOLIT/data")

df = pd.read_csv(
    DATA_DIR / "bgg_top10000.csv",
    sep=",",
    quotechar='"',
    encoding="utf-8-sig",
    engine="python",
    on_bad_lines="skip",
)
print(f"[원본] {len(df):,}행")
print(f"[컬럼] {df.columns.tolist()}")


# ── 2. rank_all → category_rank 파싱 ─────────────────────────────────────────
def parse_category_rank(text) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {}
    result = {}
    matches = re.findall(r"([A-Za-z][A-Za-z\s]*?)\s+(\d+)", text)
    for category, rank in matches:
        result[category.strip()] = int(rank)
    return result

df["category_rank"] = df["rank_all"].apply(parse_category_rank)
df = df.drop(columns=["rank_all", "num_rating"], errors="ignore")


# ── 3. players → min_players, max_players 분리 ───────────────────────────────
def parse_players(text):
    if not isinstance(text, str) or not text.strip():
        return None, None
    text = text.replace(" ", "")
    match = re.match(r"(\d+)[~\-](\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.match(r"(\d+)", text)
    if match:
        val = int(match.group(1))
        return val, val
    return None, None

df[["min_players", "max_players"]] = df["players"].apply(
    lambda x: pd.Series(parse_players(x))
)
df = df.drop(columns=["players"], errors="ignore")


# ── 4. playing_time → 숫자만 추출 ────────────────────────────────────────────
def parse_playing_time(text):
    if not isinstance(text, str):
        return None
    numbers = re.findall(r"\d+", text)
    return int(numbers[0]) if numbers else None

df["playing_time"] = df["playing_time"].apply(parse_playing_time)


# ── 5. age → 숫자만 추출 ─────────────────────────────────────────────────────
def parse_age(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None

df["age"] = df["age"].apply(parse_age)


# ── 6. 리스트형 컬럼 파싱 ────────────────────────────────────────────────────
list_columns = ["designer", "artist", "awards", "type", "category", "mechanism"]

def parse_list_column(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return [item.strip() for item in text.split(",") if item.strip()]

for col in list_columns:
    if col in df.columns:
        df[col] = df[col].apply(parse_list_column)


# ── 7. 임베딩용 chunk 생성 ───────────────────────────────────────────────────
# description → text / 나머지 → metadata
chunks = []

for _, row in df.iterrows():
    description = row.get("description", "")
    if not isinstance(description, str) or str(description) == "nan":
        description = " "
    description = description.strip() or " "  # 빈 문자열도 공백으로

    chunks.append({
        "text": description,   # 임베딩 대상
        "metadata": {
            "source":               "bgg",
            "type":                 "stats",
            "rank":                 int(row["rank"]) if pd.notna(row.get("rank")) else None,
            "title":                row.get("title"),
            "category_rank":        row.get("category_rank", {}),
            "min_players":          int(row["min_players"]) if pd.notna(row.get("min_players")) else None,
            "max_players":          int(row["max_players"]) if pd.notna(row.get("max_players")) else None,
            "recommended_players":  row.get("recommended_players"),
            "playing_time":         int(row["playing_time"]) if pd.notna(row.get("playing_time")) else None,
            "age":                  int(row["age"]) if pd.notna(row.get("age")) else None,
            "weight":               float(row["weight"]) if pd.notna(row.get("weight")) else None,
            "designer":             row.get("designer", []),
            "artist":               row.get("artist", []),
            "awards":               row.get("awards", []),
            "type":                 row.get("type", []),
            "category":             row.get("category", []),
            "mechanism":            row.get("mechanism", []),
            "image":                row.get("image"),
            "avg_rating":           float(row["avg_rating"]) if pd.notna(row.get("avg_rating")) else None,
        }
    })

print(f"[청크 생성] {len(chunks):,}개")


# ── 8. 저장 ──────────────────────────────────────────────────────────────────
# JSON
json_path = DATA_DIR / "bgg_stats_final.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print(f"JSON 저장 완료 → {json_path}")

# CSV (metadata는 JSON 문자열로 변환)
csv_path = DATA_DIR / "bgg_stats_final.csv"
csv_rows = [{"text": c["text"], "metadata": json.dumps(c["metadata"], ensure_ascii=False)} for c in chunks]
pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)
print(f"CSV 저장 완료 → {csv_path}")