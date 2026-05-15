import pandas as pd
import glob
import os
from pathlib import Path


# ── 설정 ────────────────────────────────────────────
DATA_DIR    = Path("C:/lecture/NOLIT/data")
CSV_OUTPUT  = DATA_DIR / "bgg_reviews_merged.csv"
JSON_OUTPUT = DATA_DIR / "bgg_reviews_merged.json"


# ── 1. cleaned 파일 전부 수집 ────────────────────────
files = sorted(DATA_DIR.glob("bgg_reviews_*_cleaned.csv"))

print(f"찾은 파일 수: {len(files)}개")
for f in files:
    print(f"  └ {f.name}")

if not files:
    print("파일을 찾을 수 없어요! 경로 확인해주세요.")
    exit()


# ── 2. 읽어서 합치기 ─────────────────────────────────
dfs = []
for f in files:
    df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
    print(f"{f.name}: {len(df):,}행")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
print(f"\n[합산] {len(merged):,}행")


# ── 3. 타입 정리 ─────────────────────────────────────
# rank → 정수 변환 (섞인 타입 해결)
merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce").astype("Int64")

# rating 빈칸 → None
merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")

none_count = merged["rating"].isna().sum()
print(f"[rating None 처리] {none_count:,}개")


# ── 4. rank 기준 정렬 ────────────────────────────────
merged = merged.sort_values("rank").reset_index(drop=True)
print(f"[rank 정렬 완료]")


# ── 5. 중복 제거 ─────────────────────────────────────
before = len(merged)
merged = merged.drop_duplicates().reset_index(drop=True)
print(f"[중복 제거] {before - len(merged):,}행 제거 → 최종 {len(merged):,}행")


# ── 6. 저장 ──────────────────────────────────────────
# CSV
merged.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nCSV 저장 완료 → {CSV_OUTPUT}")

# JSON
merged.to_json(JSON_OUTPUT, orient="records", force_ascii=False, indent=2)
print(f"JSON 저장 완료 → {JSON_OUTPUT}")