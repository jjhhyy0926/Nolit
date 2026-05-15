"""
murdermysterylog.com 크롤러 (수정본)
- 대상: https://murdermysterylog.com/theme/
- 추출: 게임이름 / 별점 / 인원 / 시간 / 게임설명 / 태그 / 리뷰(내용 있는 것만)
- 저장: games.json

설치:
    pip install selenium beautifulsoup4 webdriver-manager
"""

import json
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# ── 설정 ──────────────────────────────────────────────
BASE_URL  = "https://murdermysterylog.com"
LIST_URL  = f"{BASE_URL}/theme/"
OUTPUT    = "games.json"
HEADLESS   = False    # False 로 바꾸면 브라우저 창이 열립니다
WAIT_SEC   = 10      # 요소 대기 최대 시간(초)
DELAY      = 1.5     # 페이지 이동 후 안정화 대기(초)
# ──────────────────────────────────────────────────────


def make_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def safe_click(driver, element):
    try:
        element.click()
    except ElementClickInterceptedException:
        driver.execute_script("arguments[0].click();", element)


def get_game_links(driver):
    """목록 페이지 무한 스크롤하며 상세 URL 수집"""
    driver.get(LIST_URL)
    time.sleep(DELAY)

    links = set()
    prev_count = -1

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

        for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='/detail/']"):
            href = a.get_attribute("href")
            if href:
                links.add(href)

        if len(links) == prev_count:
            break
        prev_count = len(links)

    print(f"[목록] {len(links)}개 링크 수집 완료")
    return sorted(links)


def expand_description(driver):
    """더보기 버튼 클릭"""
    try:
        btn = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//*[contains(text(),'더보기') or contains(text(),'더 보기')]")
            )
        )
        safe_click(driver, btn)
        time.sleep(0.5)
    except TimeoutException:
        pass


def parse_detail(driver, url):
    driver.get(url)
    time.sleep(DELAY)
    expand_description(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # ── 게임 이름 ──────────────────────────────────────
    # h1 > div.fonttest (bold) 만 가져옴 — sub header(출판사) 제외
    name = ""
    h1 = soup.select_one("h1.ui.dividing.header")
    if h1:
        # sub header 제거 후 첫 번째 div.fonttest만 추출
        for sub in h1.select(".sub.header"):
            sub.decompose()
        name_el = h1.select_one("div.fonttest")
        if name_el:
            name = name_el.get_text(strip=True)

    # ── 별점 ───────────────────────────────────────────
    # <div class="ui blue big label"> <i class="yellow star icon"> 3.3 </div>
    rating = ""
    label_el = soup.select_one("div.ui.blue.big.label")
    if label_el:
        # i 태그 제거 후 텍스트만 추출
        for i in label_el.select("i"):
            i.decompose()
        txt = label_el.get_text(strip=True)
        m = re.search(r"\d+\.?\d*", txt)
        if m:
            rating = m.group()

    # ── 인원 ───────────────────────────────────────────
    players = ""
    page_text = soup.get_text(" ", strip=True)
    m = re.search(r"(\d+\s*[~～]\s*\d+)\s*명", page_text)
    if m:
        players = m.group(0)
    else:
        m = re.search(r"(\d+)\s*명", page_text)
        if m:
            players = m.group(0)

    # ── 소요 시간 ──────────────────────────────────────
    play_time = ""
    m = re.search(r"(\d+)\s*분", page_text)
    if m:
        play_time = m.group(0)

    # ── 게임 설명 ──────────────────────────────────────
    # style에 cursor: default, max-height 포함된 div
    description = ""
    for div in soup.select("div[style*='cursor: default'][style*='max-height']"):
        txt = div.get_text("\n", strip=True)
        if len(txt) > 20:
            description = txt
            break

    # ── 태그 (시리즈 / 제작 / 출판사 / 국내 출판사) ──────
    # 시리즈·제작·국내출판사 → 단일값, 출판사 → 중복 제거 후 복수 허용
    tags = {}
    SINGLE_KEYS = {"시리즈", "제작", "국내 출판사", "국내출판사"}
    publisher_seen = []  # 출판사 중복 제거용

    labels_wrap = soup.select_one("div.ui.labels")
    if labels_wrap:
        for a in labels_wrap.select("a"):
            txt = a.get_text(strip=True)
            for key in ["시리즈", "제작", "출판사", "국내 출판사", "국내출판사"]:
                if txt.startswith(f"{key}:"):
                    value = txt.split(":", 1)[-1].strip()

                    if key in SINGLE_KEYS:
                        # 단일값: 처음 나온 것만 저장
                        if key not in tags:
                            tags[key] = value

                    else:
                        # 출판사: 중복 제거하면서 복수 허용
                        if value not in publisher_seen:
                            publisher_seen.append(value)

                    break

    # 출판사 정리: 1개면 문자열, 2개 이상이면 리스트
    if len(publisher_seen) == 1:
        tags["출판사"] = publisher_seen[0]
    elif len(publisher_seen) > 1:
        tags["출판사"] = publisher_seen

    # ── 리뷰 (body만, 내용 있는 것만) ─────────────────
    reviews = []
    for desc_div in soup.select("div.description"):
        p = desc_div.select_one("p.fonttest")
        if not p:
            continue
        body = p.get_text(strip=True)
        if not body:
            continue

        reviews.append(body)

    return {
        "url": url,
        "name": name,
        "rating": rating,
        "players": players,
        "play_time": play_time,
        "description": description,
        "tags": tags,
        "reviews": reviews,   # 리뷰 본문 문자열 리스트
    }


def crawl_all():
    driver = make_driver(HEADLESS)
    results = []

    try:
        links = get_game_links(driver)

        for i, url in enumerate(links, 1):
            print(f"[{i}/{len(links)}] {url}")
            try:
                data = parse_detail(driver, url)
                results.append(data)
                print(f"  ✓ {data['name']} | ★{data['rating']} | {data['players']} | {data['play_time']} | 리뷰 {len(data['reviews'])}개")
            except Exception as e:
                print(f"  ✗ 오류: {e}")
                results.append({"url": url, "error": str(e)})

    finally:
        driver.quit()

    return results


def main():
    print("=" * 55)
    print("  murdermysterylog 크롤러 시작")
    print("=" * 55)

    mudermysterylog = crawl_all()

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(mudermysterylog, f, ensure_ascii=False, indent=2)

    print(f"\n완료! {len(mudermysterylog)}개 게임 → {OUTPUT} 저장")


if __name__ == "__main__":
    main()