import requests
import pandas as pd

API_URL = "https://q.keigon.net/indexes/qrooms/search"
TOKEN   = "F3WdGD8S3783e99ba7d4508fa06c0dc6d1822e2bd73b87bfe2dd204eae14eb34220cda75"

bba_headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Origin": "https://bbabang.net",
    "Referer": "https://bbabang.net/",
}

# 전체 테마 수집
print("테마 데이터 수집 중...")
all_hits = []
offset = 0
while True:
    payload = {"q": "", "limit": 1000, "offset": offset}
    resp = requests.post(API_URL, headers=bba_headers, json=payload, timeout=15)
    data = resp.json()
    hits = data.get("hits", [])
    if not hits:
        break
    all_hits.extend(hits)
    total = data.get("estimatedTotalHits", 0)
    print(f"  수집: {len(all_hits)} / {total}")
    if len(all_hits) >= total or len(hits) < 1000:
        break
    offset += 1000

print(f"총 {len(all_hits)}개 테마 수집 완료")

# 필요한 컬럼만 추출
rows = []
for h in all_hits:
    rows.append({
        "ref_id":           h.get("ref_id", ""),
        "title":            h.get("title", ""),
        "store_name":       h.get("store_name", ""),
        "address":          h.get("address", ""),
        "location":         h.get("location", ""),
        "area":             h.get("area", ""),
        "description":      h.get("description", "").strip() if h.get("description") else "",
        "playing_time":     h.get("playtime", ""),
        "player_count":     h.get("player_count", ""),
        "price":            h.get("price", ""),
        "difficulty":       round(h.get("difficultyTotalRating", 0), 2),
        "horror":           round(h.get("fearTotalRating", 0), 2),
        "activity":         round(h.get("activityTotalRating", 0), 2),
        "satisfaction":     round(h.get("satisfyTotalRating", 0), 2),
        "puzzle":           round(h.get("problemTotalRating", 0), 2),
        "story":            round(h.get("storyTotalRating", 0), 2),
        "interior":         round(h.get("interiorTotalRating", 0), 2),
        "production":       round(h.get("actTotalRating", 0), 2),
        "review_count":     h.get("reviewCount", 0),
        "recommend_count":  h.get("recommendReviewCount", 0),
        "tags":             ", ".join(h.get("tags", [])),
        "special_tags":     ", ".join(h.get("special_tags", [])),
        "store_homepage":   h.get("store_homepage", ""),
        "store_tel":        h.get("store_tel", ""),
        "isopen":           h.get("isopen", ""),
        "lat":              h.get("_geo", {}).get("lat", ""),
        "lng":              h.get("_geo", {}).get("lng", ""),
    })

df = pd.DataFrame(rows)

print("\n=== 미리보기 ===")
print(df[["title", "store_name", "satisfaction", "horror", "difficulty", "review_count"]].head(10).to_string())
print(f"\n총 {len(df)}개 테마")

df.to_csv("bbabang_themes.csv", index=False, encoding="utf-8-sig")
print("저장 완료: bbabang_themes.csv")