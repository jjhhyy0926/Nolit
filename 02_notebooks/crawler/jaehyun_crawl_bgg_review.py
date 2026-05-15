import subprocess
import sys

# 필요한 패키지 자동 설치
subprocess.run([sys.executable, "-m", "pip", "install", "playwright", "beautifulsoup4", "pandas", "lxml"], check=True)
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)

import re, time, os, pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BGG_BASE = "https://boardgamegeek.com"
DELAY    = 2

# ──────────────────────────────────────────────
# ★ 설정값 - 여기만 수정하세요 ★
# ──────────────────────────────────────────────
BGG_ID     = "flow0003"   # BGG 아이디
BGG_PW     = "skn26_01"   # BGG 비밀번호
START_RANK = 1501
END_RANK   = 2000
MAX_PAGES  = 20

CSV_NAME   = f"bgg_reviews_{START_RANK}_{END_RANK}.csv"

# ──────────────────────────────────────────────
# 로그인 (쿠키 동의 + 팝업 방식)
# ──────────────────────────────────────────────
def get_login_cookies():
    print("BGG 로그인 중...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        page.goto("https://boardgamegeek.com", wait_until="domcontentloaded")
        time.sleep(3)

        # 쿠키 동의 팝업 닫기
        for selector in [
            "button:has-text(\"I'm OK with that\")",
            'button:has-text("Accept All")',
            'button:has-text("Accept")',
            '.c-bn',
        ]:
            try:
                page.click(selector, timeout=5000)
                time.sleep(1)
                print("  쿠키 동의 완료")
                break
            except:
                continue
        else:
            print("  쿠키 팝업 없음, 계속 진행")

        # Sign In 클릭
        page.click('a:has-text("Sign In")')
        time.sleep(2)

        page.fill('#inputUsername', BGG_ID)
        time.sleep(1)
        page.fill('#inputPassword', BGG_PW)
        time.sleep(1)

        try:
            page.click('button:has-text("Sign In")')
        except:
            page.keyboard.press("Enter")

        time.sleep(5)
        print(f"  로그인 후 URL: {page.url}")

        soup = BeautifulSoup(page.content(), "lxml")
        if soup.find("a", href=re.compile(r'/user/')):
            print(f"  ✅ 로그인 성공!")
        else:
            print(f"  ⚠️ 로그인 확인 불분명, 계속 진행합니다...")

        cookies = context.cookies()
        browser.close()
        print(f"  쿠키 {len(cookies)}개 저장")
        return cookies

# ──────────────────────────────────────────────
# HTML 가져오기 (재시도 포함)
# ──────────────────────────────────────────────
def get_html(url, cookies=None, sleep_sec=5, max_retry=3):
    print(f"  fetching: {url}")
    for attempt in range(1, max_retry + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}
                )
                if cookies:
                    context.add_cookies(cookies)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                time.sleep(sleep_sec)
                html = page.content()
                browser.close()
            print(f"  done: {len(html)} chars")
            return html
        except Exception as e:
            print(f"  ⚠️ attempt {attempt}/{max_retry} 실패: {e}")
            if attempt < max_retry:
                wait = attempt * 10
                print(f"  {wait}초 후 재시도...")
                time.sleep(wait)
            else:
                print(f"  ❌ {max_retry}번 모두 실패, 스킵")
                return ""
    return ""

# ──────────────────────────────────────────────
# STEP 1: 랭킹 목록 수집
# ──────────────────────────────────────────────
def get_games_in_range(start_rank, end_rank, cookies):
    print(f"\n[STEP1] get ranking list ({start_rank}~{end_rank})...")
    games    = []
    page_num = max(1, (start_rank - 1) // 100 + 1)

    while True:
        url  = f"{BGG_BASE}/browse/boardgame/page/{page_num}"
        html = get_html(url, cookies=cookies, sleep_sec=4)
        if not html:
            page_num += 1
            continue

        soup = BeautifulSoup(html, "lxml")
        rows = soup.select("table tr")

        found      = 0
        page_ranks = []

        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 3:
                continue
            rank_text = tds[0].get_text(strip=True)
            if not re.match(r"^\d+$", rank_text):
                continue
            rank = int(rank_text)
            page_ranks.append(rank)

            if rank < start_rank or rank > end_rank:
                continue

            game_a = None
            for td in tds:
                a = td.find("a", href=re.compile(r"^/boardgame/\d+/[^/]+$"))
                if a and a.get_text(strip=True):
                    game_a = a
                    break
            if not game_a:
                continue

            href    = game_a["href"]
            parts   = href.split("/")
            if len(parts) < 4:
                continue
            game_id = parts[2]
            raw     = game_a.get_text(strip=True)
            name    = re.sub(r'\s*(Published|Upcoming|Announced)?\s*\d{4,}.*', '', raw).strip()

            if not name or any(g["game_id"] == game_id for g in games):
                continue

            games.append({
                "rank": rank, "name": name,
                "game_id": game_id, "url": BGG_BASE + href
            })
            found += 1

        page_min = min(page_ranks) if page_ranks else "?"
        page_max = max(page_ranks) if page_ranks else "?"
        print(f"  page {page_num}: {found}개 (누적 {len(games)}개) | 랭킹: {page_min}~{page_max}")

        if page_ranks and max(page_ranks) >= end_rank:
            break
        if not page_ranks:
            print(f"  빈 페이지, 종료")
            break

        page_num += 1
        time.sleep(DELAY)

    print(f"✅ 총 {len(games)}개 수집")
    return games

# ──────────────────────────────────────────────
# STEP 2: 리뷰 파싱
# ──────────────────────────────────────────────
def extract_reviews_from_html(html):
    if not html:
        return []
    soup    = BeautifulSoup(html, "lxml")
    reviews = []

    for item in soup.find_all(attrs={"ng-repeat": "item in ratingsctrl.data.items"}):
        rating_el = item.find(class_=re.compile(r'rating-angular'))
        rating    = rating_el.get_text(strip=True) if rating_el else ""

        user_el  = item.find(class_="comment-header-user")
        username = user_el.get_text(strip=True) if user_el else ""

        body_el = item.find(class_="comment-body")
        if body_el:
            text = body_el.get_text(" ", strip=True)
            text = re.sub(r'\+\s*More\s*-?\s*Less', '', text, flags=re.IGNORECASE)
            text = re.sub(r'Collections?:.*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = ""

        if text and len(text) > 5:
            reviews.append({
                "username": username,
                "rating":   rating,
                "review":   text,
            })

    return reviews


def get_all_reviews(game_url, cookies, max_pages=None):
    all_reviews  = []
    base_url     = game_url.rstrip("/") + "/ratings?comment=1"
    pageid       = 1
    prev_reviews = None

    while True:
        if max_pages and pageid > max_pages:
            print(f"    max pages ({max_pages}) reached, stop")
            break

        url          = f"{base_url}&pageid={pageid}"
        html         = get_html(url, cookies=cookies, sleep_sec=4)
        page_reviews = extract_reviews_from_html(html)

        if not page_reviews:
            print(f"    pageid={pageid}: no reviews, stop")
            break

        prev_texts = [r["review"] for r in prev_reviews] if prev_reviews else []
        curr_texts = [r["review"] for r in page_reviews]
        if curr_texts == prev_texts:
            print(f"    pageid={pageid}: same as previous, stop")
            break

        all_reviews.extend(page_reviews)
        print(f"    pageid={pageid}: {len(page_reviews)}개 (누적 {len(all_reviews)}개)")
        prev_reviews = page_reviews[:]

        pageid += 1
        time.sleep(DELAY)

    return all_reviews

# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
print("="*50)
print(f"START BGG REVIEW CRAWLER")
print(f"범위: {START_RANK}위 ~ {END_RANK}위")
print(f"저장: {CSV_NAME}")
print("="*50)

# 로그인
cookies = get_login_cookies()

# 이미 수집된 데이터 있으면 이어서 수집
csv_path      = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_NAME)
done_titles   = set()
existing_rows = []

if os.path.exists(csv_path):
    df_existing   = pd.read_csv(csv_path, encoding="utf-8-sig")
    done_titles   = set(df_existing["title"].unique())
    existing_rows = df_existing.to_dict("records")
    print(f"\n기존 파일 발견: {len(existing_rows)}개 리뷰, {len(done_titles)}개 게임 완료")
    print("이어서 수집합니다...")

# 랭킹 목록 수집
games = get_games_in_range(START_RANK, END_RANK, cookies)

print(f"\n[STEP2] collect reviews ({len(games)}개 게임)...")
all_rows = existing_rows[:]

for i, g in enumerate(games):
    if g["name"] in done_titles:
        print(f"  [{g['rank']:>5}] {g['name']} → 스킵")
        continue

    print(f"\n[{g['rank']:>5}/{END_RANK}] {g['name']}  ({i+1}/{len(games)})")

    try:
        reviews = get_all_reviews(g["url"], cookies, max_pages=MAX_PAGES)
        for r in reviews:
            all_rows.append({
                "rank":     g["rank"],
                "title":    g["name"],
                "username": r["username"],
                "rating":   r["rating"],
                "review":   r["review"],
            })
        print(f"  ✅ {len(reviews)}개 리뷰 수집")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")

    # 10개 게임마다 중간 저장
    if (i + 1) % 10 == 0:
        df_tmp = pd.DataFrame(all_rows, columns=["rank", "title", "username", "rating", "review"])
        df_tmp.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  💾 중간 저장: {len(all_rows)}개 리뷰 → {csv_path}\n")

# 최종 저장
df = pd.DataFrame(all_rows, columns=["rank", "title", "username", "rating", "review"])
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("\n" + "="*50)
print(f"DONE: {len(df)}개 리뷰 저장")
print(f"  {csv_path}")
print("="*50)
print("\n=== 게임별 리뷰 수 (상위 10개) ===")
print(df.groupby(["rank", "title"])["review"].count().head(10).to_string())