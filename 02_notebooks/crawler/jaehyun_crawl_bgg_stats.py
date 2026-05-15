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
BGG_ID   = "flow0011"   # BGG 아이디
BGG_PW   = "skn26_01"
START_N  = 2451              # 시작 순위
END_N    = 2501              # 끝 순위
CSV_NAME = f"bgg_top_{START_N}_{END_N}.csv"

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
# STEP 1: 랭킹 목록 수집 (범위 지정)
# ──────────────────────────────────────────────
def get_top_games(start_n, end_n, cookies):
    print(f"\n[STEP1] get ranking list ({start_n}~{end_n})...")
    games    = []
    page_num = max(1, (start_n - 1) // 100 + 1)

    while len(games) < (end_n - start_n + 1):
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

            if rank < start_n or rank > end_n:
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

        if page_ranks and max(page_ranks) >= end_n:
            break
        if not page_ranks:
            print(f"  빈 페이지, 종료")
            break

        page_num += 1
        time.sleep(DELAY)

    print(f"✅ 총 {len(games)}개 수집")
    return games

# ──────────────────────────────────────────────
# STEP 2: 메인 페이지 파싱
# ──────────────────────────────────────────────
def parse_game_page(html):
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    m = re.search(r'(\d+)\s*(?:–|-)\s*(\d+)\s*Players', text)
    players = f"{m.group(1)}~{m.group(2)}" if m else ""

    m = re.search(r'Best:\s*([\d\-~]+)', text)
    if not m:
        m = re.search(r'Best:\s*(\d+)', text)
    recommended_players = m.group(1) if m else ""

    m = re.search(r'(\d+)\s*(?:–|-)\s*(\d+)\s*Min\s*Playing Time', text)
    if m:
        playing_time = f"{m.group(2)}min"
    else:
        m = re.search(r'(\d+)\s*Min\s*Playing Time', text)
        playing_time = f"{m.group(1)}min" if m else ""

    m = re.search(r'Age:\s*(\d+)\s*\+', text)
    age = f"{m.group(1)}+" if m else ""

    weight = ""
    for pattern in [
        r'Weight:\s*[\u2013\u2014\-\s]*([\d.]+)\s*/\s*5',
        r'([\d.]+)\s*/\s*5\s*Complexity',
        r'Complexity[^\d]*([\d.]+)\s*/\s*5',
        r'Weight[^\d]*([\d.]+)\s*/\s*5',
    ]:
        m = re.search(pattern, text)
        if m:
            weight = m.group(1)
            break

    description = ""
    for cls in ["game-description-body", "game-description"]:
        desc_el = soup.find(class_=cls)
        if desc_el:
            description = desc_el.get_text(" ", strip=True)
            break
    if not description:
        desc_el = soup.find(attrs={"ng-bind-html": "geekitemctrl.wikitext|to_trusted"})
        if desc_el:
            description = desc_el.get_text(" ", strip=True)
    if not description:
        article = soup.find("article")
        if article:
            description = article.get_text(" ", strip=True)
    description = re.sub(r'\s+', ' ', description).strip()

    og = soup.find("meta", attrs={"property": "og:image"})
    image = og.get("content", "") if og else ""

    def get_links(keyword):
        return list(dict.fromkeys(
            a.get_text(strip=True) for a in soup.find_all("a", href=True)
            if keyword in a["href"] and a.get_text(strip=True)
        ))

    return {
        "players":             players,
        "recommended_players": recommended_players,
        "playing_time":        playing_time,
        "age":                 age,
        "weight":              weight,
        "designer":            ", ".join(get_links("/boardgamedesigner/")),
        "artist":              ", ".join(get_links("/boardgameartist/")),
        "description":         description,
        "awards":              ", ".join(get_links("/boardgamehonor/")),
        "type":                ", ".join(get_links("/boardgamesubdomain/")),
        "image":               image,
    }

# ──────────────────────────────────────────────
# STEP 3: /credits
# ──────────────────────────────────────────────
def parse_credits_page(html):
    if not html:
        return {"category": "", "mechanism": ""}
    soup = BeautifulSoup(html, "lxml")
    categories = list(dict.fromkeys(
        a.get_text(strip=True) for a in soup.find_all("a", href=True)
        if "/boardgamecategory/" in a["href"] and a.get_text(strip=True)
    ))
    mechanisms = list(dict.fromkeys(
        a.get_text(strip=True) for a in soup.find_all("a", href=True)
        if "/boardgamemechanic/" in a["href"] and a.get_text(strip=True)
    ))
    return {"category": ", ".join(categories), "mechanism": ", ".join(mechanisms)}

# ──────────────────────────────────────────────
# STEP 4: /stats
# ──────────────────────────────────────────────
def parse_stats_page(html):
    if not html:
        return {"avg_rating": "", "num_rating": "", "rank_all": ""}
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    m = re.search(r'Avg\.?\s*Rating\s*([\d.]+)', text)
    avg_rating = m.group(1) if m else ""

    m = re.search(r'No\.?\s*of\s*Ratings\s*([\d,]+)', text)
    num_rating = m.group(1) if m else ""

    rank_map = {
        "rankobjectid=1":    "Overall",
        "rankobjectid=5497": "Strategy",
        "rankobjectid=5499": "Family",
        "rankobjectid=5496": "Thematic",
        "rankobjectid=4666": "Abstract",
        "rankobjectid=5498": "Party",
        "rankobjectid=4665": "Childrens",
        "rankobjectid=4664": "War",
    }
    rank_parts = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href  = a["href"]
        label = a.get_text(strip=True)
        if not label or not re.match(r"^\d+$", label):
            continue
        for key, name in rank_map.items():
            if key in href and name not in seen:
                rank_parts.append(f"{name} {label}")
                seen.add(name)
                break

    return {
        "avg_rating": avg_rating,
        "num_rating": num_rating,
        "rank_all":   ", ".join(rank_parts),
    }

# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
print("="*50)
print(f"START BGG CRAWLER")
print(f"범위: {START_N}위 ~ {END_N}위")
print("="*50)

# 로그인
cookies = get_login_cookies()

# 이미 수집된 데이터 확인
csv_path    = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_NAME)
done_titles = set()
results     = []

if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path, encoding="utf-8-sig")
    done_titles = set(df_existing["title"].unique())
    results     = df_existing.to_dict("records")
    print(f"\n기존 파일 발견: {len(results)}개 완료, 이어서 수집...")
else:
    print(f"\n새로 시작합니다...")

# 랭킹 목록 수집
top_games = get_top_games(START_N, END_N, cookies)

empty_details = {
    "players":"", "recommended_players":"", "playing_time":"", "age":"",
    "weight":"", "designer":"", "artist":"", "description":"", "awards":"",
    "type":"", "image":""
}
empty_credits = {"category":"", "mechanism":""}
empty_stats   = {"avg_rating":"", "num_rating":"", "rank_all":""}

col_order = [
    "rank", "rank_all", "title", "players", "recommended_players",
    "playing_time", "age", "weight", "designer", "artist",
    "description", "awards", "type", "category", "mechanism",
    "image", "avg_rating", "num_rating", "bgg_url",
]

print(f"\n[STEP2] crawl details + credits + stats ({len(top_games)}개 게임)...")

for i, g in enumerate(top_games):
    if g["name"] in done_titles:
        print(f"  [{g['rank']:>5}] {g['name']} → 스킵")
        continue

    print(f"\n[{g['rank']:>5}/{END_N}] {g['name']}  ({i+1}/{len(top_games)})")

    try:
        details = parse_game_page(get_html(g["url"], cookies=cookies, sleep_sec=5))
        print(f"  ✅ players={details.get('players','')}  time={details.get('playing_time','')}  weight={details.get('weight','')}")
    except Exception as e:
        print(f"  ❌ detail: {e}")
        details = empty_details.copy()
    time.sleep(DELAY)

    try:
        credits = parse_credits_page(get_html(g["url"].rstrip("/") + "/credits", cookies=cookies, sleep_sec=4))
        print(f"  ✅ cat={len(credits['category'].split(','))}개  mech={len(credits['mechanism'].split(','))}개")
    except Exception as e:
        print(f"  ❌ credits: {e}")
        credits = empty_credits.copy()
    time.sleep(DELAY)

    try:
        stats = parse_stats_page(get_html(g["url"].rstrip("/") + "/stats", cookies=cookies, sleep_sec=4))
        print(f"  ✅ avg={stats['avg_rating']}  num={stats['num_rating']}  rank={stats['rank_all']}")
    except Exception as e:
        print(f"  ❌ stats: {e}")
        stats = empty_stats.copy()
    time.sleep(DELAY)

    results.append({
        "rank":    g["rank"],
        "title":   g["name"],
        "bgg_url": g["url"],
        **details,
        **credits,
        **stats,
    })

    # 50개마다 중간 저장
    if len(results) % 50 == 0:
        df_tmp = pd.DataFrame(results)
        df_tmp = df_tmp[[c for c in col_order if c in df_tmp.columns]]
        df_tmp.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  💾 중간 저장: {len(results)}개 → {csv_path}\n")

# 최종 저장
df = pd.DataFrame(results)
df = df[[c for c in col_order if c in df.columns]]
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("\n" + "="*50)
print(f"DONE: {len(df)}개 게임 저장")
print(f"  {csv_path}")
print("="*50)
print(df[["rank", "title", "players", "playing_time", "weight", "avg_rating"]].head(10).to_string())