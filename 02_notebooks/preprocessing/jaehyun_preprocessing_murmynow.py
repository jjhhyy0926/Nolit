import json
import re
import csv
import pandas as pd
from pathlib import Path

# ── 설정 ────────────────────────────────────────────
DATA_DIR    = Path("C:/lecture/NOLIT/data")
INPUT_PATH  = DATA_DIR / "murmynow.json"
JSON_OUTPUT = DATA_DIR / "mummynow_chunks.json"
CSV_OUTPUT  = DATA_DIR / "mummynow_chunks.csv"

RECORD_PATTERNS = [
    r"^\d{4}\.\d{2}\.\d{2}$",
    r"^\d{4}\.\d{2}\.\d{2}\s*플레이.*$",
    r"^\d{1,2}회차$",
    r"^플레이\s*(완료|함|했음|기록)$",
    r"^기록용$",
    r"^\d{1,2}회차\s*기록$",
]

DIFFICULTY_MAP = {
    "쉬워요":        1,
    "보통이에요":    2,
    "어려워요":      3,
    "매우 어려워요": 4,
}

# ── 유틸 함수 ────────────────────────────────────────

def clean_value(val):
    if val is None:
        return None
    val = str(val).strip()
    if val in ["미정", "", "-", "없음", "None", "nan"]:
        return None
    return val

def parse_play_time(val):
    val = clean_value(val)
    if val is None:
        return None, None
    val = val.replace("+", "").replace("분", "").strip()
    if "~" in val:
        parts = val.split("~")
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except:
            return None, None
    try:
        t = int(val)
        return t, t
    except:
        return None, None

def parse_players(val):
    val = clean_value(val)
    if val is None:
        return None, None
    val = val.replace("인", "").replace("명", "").strip()
    if "~" in val:
        parts = val.split("~")
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except:
            return None, None
    try:
        n = int(val)
        return n, n
    except:
        return None, None

def parse_difficulty(val):
    val = clean_value(val)
    if val is None:
        return None
    if val in DIFFICULTY_MAP:
        return DIFFICULTY_MAP[val]
    try:
        return float(val)
    except:
        return None

def is_record_review(text):
    if text is None:
        return True
    text = text.strip()
    if text == "":
        return True
    for pattern in RECORD_PATTERNS:
        if re.match(pattern, text):
            return True
    return False

# ── 메인 처리 ────────────────────────────────────────

def process_mummynow(input_path, json_output, csv_output):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"원본 데이터: {len(data)}행")

    # ── 스탯 chunk ───────────────────────────────────
    seen_names   = set()
    stats_chunks = []

    for item in data:
        name = clean_value(item.get("name"))
        if name is None or name in seen_names:
            continue
        seen_names.add(name)

        min_p, max_p = parse_players(item.get("players"))
        min_t, max_t = parse_play_time(item.get("play_time"))
        difficulty   = parse_difficulty(item.get("difficulty"))

        rating = None
        r = clean_value(item.get("rating"))
        if r:
            try:
                rating = float(r)
            except:
                pass

        description = clean_value(item.get("description"))
        if description:
            description = description.replace("\r\n", " ").replace("\n", " ").strip()

        # description 없으면 스탯 chunk 생성 안 함
        if not description:
            continue

        text = description

        stats_chunks.append({
            "text": text,
            "metadata": {
                "source":         "mummynow",
                "type":           "stats",
                "name":           name,
                "scene_category": clean_value(item.get("category")),
                "author":         clean_value(item.get("author")),
                "publisher":      clean_value(item.get("publisher")),
                "rating":         rating,
                "difficulty":     difficulty,
                "min_players":    min_p,
                "max_players":    max_p,
                "min_time":       min_t,
                "max_time":       max_t,
                "image_url":      clean_value(item.get("image_url")),
            }
        })

    print(f"스탯 chunk: {len(stats_chunks)}개")

    # ── 리뷰 chunk ───────────────────────────────────
    review_chunks  = []
    removed_record = 0
    removed_empty  = 0

    for item in data:
        name        = clean_value(item.get("name"))
        review_text = clean_value(item.get("review_text"))

        if review_text is None:
            removed_empty += 1
            continue
        if is_record_review(review_text):
            removed_record += 1
            continue

        review_difficulty = parse_difficulty(item.get("review_difficulty"))

        review_rating = None
        rr = clean_value(item.get("review_rating"))
        if rr:
            try:
                review_rating = float(rr)
            except:
                pass

        review_chunks.append({
            "text": review_text,   # 리뷰 텍스트만 임베딩 대상
            "metadata": {
                "source":            "mummynow",
                "type":              "review",
                "name":              name,
                "review_rating":     review_rating,
                "review_difficulty": review_difficulty,
            }
        })

    print(f"리뷰 chunk: {len(review_chunks)}개")
    print(f"  └ 빈 리뷰 제거:     {removed_empty}개")
    print(f"  └ 기록용 리뷰 제거: {removed_record}개")

    # ── 합치기 및 저장 ───────────────────────────────
    all_chunks = stats_chunks + review_chunks

    # JSON 저장
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 저장 완료 → {json_output}")

    # CSV 저장
    csv_rows = [
        {"text": c["text"], "metadata": json.dumps(c["metadata"], ensure_ascii=False)}
        for c in all_chunks
    ]
    pd.DataFrame(csv_rows).to_csv(
        csv_output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"CSV 저장 완료 → {csv_output}")

    print(f"\n전체 chunk: {len(all_chunks)}개")
    return all_chunks


if __name__ == "__main__":
    process_mummynow(INPUT_PATH, JSON_OUTPUT, CSV_OUTPUT)