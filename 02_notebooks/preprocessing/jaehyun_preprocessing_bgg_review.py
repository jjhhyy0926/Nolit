import pandas as pd
import re
import csv
from pathlib import Path


# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
DATA_DIR  = Path("C:/lecture/NOLIT/data")
FILE_NAME = "bgg_reviews_6667_10000.csv"   # ← 전처리할 파일명 여기서 바꾸기

df = pd.read_csv(
    DATA_DIR / FILE_NAME,
    sep=",",
    quotechar='"',
    encoding="utf-8-sig",
    engine="python",
    on_bad_lines="skip",
)
print(f"[원본] {len(df):,}행")
print(f"[컬럼] {df.columns.tolist()}")

# Unnamed 컬럼 제거 (혹시 남아있으면)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]


# ── 2. username 컬럼 제거 ─────────────────────────────────────────────────────
df = df.drop(columns=["username"], errors="ignore")


# ── 3. review 전처리 함수 ─────────────────────────────────────────────────────

def is_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", text))

def is_english(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))

def has_enough_english_words(text: str, min_words: int = 2) -> bool:
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    return len(words) >= min_words

def is_only_special_or_number(text: str) -> bool:
    cleaned = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    return len(cleaned) == 0

def is_valid_review(text) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text:
        return False
    if is_only_special_or_number(text):
        return False
    if not is_english(text) and not is_korean(text):
        return False
    if not is_korean(text) and not has_enough_english_words(text, min_words=2):
        return False
    return True

def extract_valid_language_blocks(text: str) -> str:
    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
        elif is_english(stripped) or is_korean(stripped):
            kept.append(line)
    return "\n".join(kept).strip()


# ── 4. 전처리 적용 ────────────────────────────────────────────────────────────
df["review"] = df["review"].apply(
    lambda x: extract_valid_language_blocks(x) if isinstance(x, str) else x
)

mask_valid = df["review"].apply(is_valid_review)
removed = (~mask_valid).sum()
df = df[mask_valid].reset_index(drop=True)

print(f"[제거됨] {removed:,}행")
print(f"[최종]   {len(df):,}행")


# ── 5. 저장 ───────────────────────────────────────────────────────────────────
out_name = FILE_NAME.replace(".csv", "_cleaned.csv")
out_path = DATA_DIR / out_name

df.to_csv(
    out_path,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_NONNUMERIC,  # 텍스트 필드 전부 따옴표로 감싸서 저장
)
print(f"저장 완료 → {out_path}")