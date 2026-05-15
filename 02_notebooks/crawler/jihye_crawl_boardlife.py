"""
boardlife.co.kr/rank 크롤러
- 1~17페이지 순위 목록 수집
- 각 게임 상세 페이지 전체 컬럼 수집
- 결과: boardlife_games.csv
"""

"""
실행
venv\Scripts\activate
python crawling.py
"""
import time
import re
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

BASE_URL = "https://boardlife.co.kr"
RANK_URL = "https://boardlife.co.kr/rank"
TOTAL_PAGES = 17
#TOTAL_PAGES = 1  # 테스트
PAGE_PAUSE = 2.0
DETAIL_PAUSE = 2.0
SCROLL_PAUSE = 1.5
SAVE_INTERVAL = 10

# 드라이버 경로 최초 1회만 다운로드
DRIVER_PATH = ChromeDriverManager().install()


def init_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    # PC 버전 User-Agent
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(service=Service(DRIVER_PATH), options=options)


def restart_driver(driver):
    try:
        driver.quit()
    except Exception:
        pass
    time.sleep(2)
    return init_driver()


def scroll_to_bottom(driver, pause=SCROLL_PAUSE):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def get_rank_list_from_page(driver, page_num):
    url = f"{RANK_URL}/all/{page_num}"
    driver.get(url)
    time.sleep(PAGE_PAUSE)
    scroll_to_bottom(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    games = []

    title_links = [
        a for a in soup.find_all("a", href=re.compile(r"^/game/\d+$"))
        if "title" in a.get("class", [])
    ]

    for i, a_tag in enumerate(title_links):
        try:
            href = BASE_URL + a_tag.get("href", "")
            title = a_tag.get_text(strip=True)
            row = a_tag.find_parent(class_="rank-row")

            rank_tag = row.select_one(".rank") if row else None
            rank_text = rank_tag.get_text(strip=True) if rank_tag else ""
            rank_num = re.sub(r"\D", "", rank_text)
            rank = int(rank_num) if rank_num else (page_num - 1) * 100 + i + 1

            img_tag = row.select_one("img") if row else None
            if img_tag:
                image = img_tag.get("data-src") or img_tag.get("src", "")
                if image and not image.startswith("http"):
                    image = BASE_URL + image
            else:
                image = ""

            if title and href:
                games.append({
                    "rank": rank,
                    "title": title,
                    "detail_url": href,
                    "image": image,
                    "avg_rating": "",
                })

        except Exception as e:
            print(f"  [오류] 항목 파싱: {e}")
            continue

    print(f"  -> p{page_num}: {len(games)}개 게임 수집")
    return games


def get_detail_info(driver, game):
    driver.get(game["detail_url"])
    time.sleep(DETAIL_PAUSE)
    scroll_to_bottom(driver, pause=1.0)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    info = {}

    # 한글 제목
    title_tag = soup.select_one("#boardgame-title")
    if title_tag:
        info["title"] = title_tag.get_text(strip=True)

    # 영문 제목
    eng_tag = soup.select_one("h2.font-17.main-color")
    if eng_tag:
        info["title_eng"] = eng_tag.get_text(strip=True)

    # 이미지
    img_tag = soup.select_one(".main-img img")
    if img_tag:
        info["image"] = img_tag.get("src", "") or img_tag.get("data-src", "")

    # 평점: dd .main-color 첫 번째
    rating_spans = soup.select("dd .main-color")
    if rating_spans:
        info["avg_rating"] = rating_spans[0].get_text(strip=True)

    # 평점 참여자 수: .community-data-title "평가 8.5 (661)"
    rate_title = soup.select_one(".community-data-title")
    if rate_title:
        match = re.search(r'\((\d[\d,]*)\)', rate_title.get_text())
        if match:
            info["num_rating"] = match.group(1).replace(",", "")

    # 난이도
    weight_tag = soup.select_one("#game-weight")
    if weight_tag:
        info["weight"] = weight_tag.get_text(strip=True)

    # 순위
    rank_tag = soup.select_one(".game-rank-num span")
    if rank_tag:
        info["rank"] = rank_tag.get_text(strip=True)

    # 인원/시간/연령: dl.flex-div.bullet
    for dl in soup.select("dl.flex-div.bullet"):
        dt = dl.select_one("dt")
        dd = dl.select_one("dd.data")
        if not dt or not dd:
            continue
        label = dt.get_text(strip=True)
        value = dd.get_text(strip=True)
        value_clean = re.sub(r'\(.*?\)', '', value).strip()
        if "인원" in label:
            info["players"] = value_clean
            rec = dd.select_one(".recommend-player")
            if rec:
                info["recommended_players"] = rec.get_text(strip=True)
        elif "시간" in label:
            info["playing_time"] = value_clean
        elif "연령" in label or "나이" in label:
            info["age"] = value_clean

    # 디자이너/아트워크: dl.credit-row.flex-div
    for dl in soup.select("dl.credit-row.flex-div"):
        dt = dl.select_one("dt.credit-title")
        dd = dl.select_one("dd.data")
        if not dt or not dd:
            continue
        label = dt.get_text(strip=True)
        val = ", ".join(a.get_text(strip=True) for a in dd.select("a"))
        if "디자이너" in label:
            info["designer"] = val
        elif "아트" in label:
            info["artist"] = val

    # 카테고리/테마/진행방식: .credits-box
    for box in soup.select(".credits-box"):
        title_info = box.select_one(".title-info")
        if not title_info:
            continue
        label = title_info.get_text(strip=True)
        items = [a.get_text(strip=True) for a in box.select(".credits-row a.title")
                 if "더보기" not in a.get_text()]
        val = ", ".join(items)
        if "카테고리" in label:
            info["type"] = val
        elif "테마" in label:
            info["category"] = val
        elif "진행방식" in label:
            info["mechanism"] = val

    # 게임 설명
    desc_tag = soup.select_one(".content.description")
    if desc_tag:
        for tag in desc_tag.select("#gradient-box, #description-btn"):
            tag.decompose()
        info["description"] = desc_tag.get_text(strip=True)

    info.setdefault("recommended_players", None)
    info.setdefault("weight", None)

    return {**game, **info}


# def get_reviews(driver, game_id):
#     reviews = []
#     page = 1
#     while True:
#         url = f"{BASE_URL}/game/{game_id}/rate?pg={page}"
#         driver.get(url)
#         time.sleep(DETAIL_PAUSE)
#         scroll_to_bottom(driver, pause=1.0)
#         soup = BeautifulSoup(driver.page_source, "html.parser")
#         rate_rows = soup.select(".board-list-box .rate-row")
#         if not rate_rows:
#             break
#         page_reviews = []
#         for row in rate_rows:
#             comment_tag = row.select_one(".rate-comment")
#             point_tag = row.select_one(".rate-point")
#             nick_tag = row.select_one(".nick")
#             date_tag = row.select_one(".date")
#             page_reviews.append({
#                 "game_id": game_id,
#                 "review_text": comment_tag.get_text(strip=True) if comment_tag else "",
#                 "rating": point_tag.get_text(strip=True) if point_tag else "",
#                 "reviewer": nick_tag.get_text(strip=True) if nick_tag else "",
#                 "date": date_tag.get_text(strip=True) if date_tag else "",
#                 "review_page": page,
#             })
#         if not page_reviews:
#             break
#         reviews.extend(page_reviews)
#         print(f"    -> 리뷰 p{page}: {len(page_reviews)}개")
#         next_link = soup.select_one(f"a[onclick*=\"'pg','{page + 1}'\"]")
#         if not next_link:
#             break
#         page += 1
#         if page > 50:
#             break
#     return reviews


def save_games(games, cols_order):
    if not games:
        return
    df = pd.DataFrame(games)
    for col in cols_order:
        if col not in df.columns:
            df[col] = None
    df = df[cols_order]
    header = not os.path.exists("boardlife_games.csv")
    df.to_csv("boardlife_games.csv", mode="a", index=False, encoding="utf-8-sig", header=header)


def extract_game_id(detail_url):
    match = re.search(r'/game/(\d+)', detail_url)
    return match.group(1) if match else None


def main():
    print("=" * 50)
    print("보드라이프 크롤러 시작")
    print(f"대상: 1~{TOTAL_PAGES}페이지")
    print("=" * 50)

    cols_order = [
        "rank", "title", "title_eng",
        "players", "recommended_players",
        "playing_time", "age", "weight",
        "designer", "artist", "description",
        "type", "category", "mechanism",
        "image", "avg_rating", "num_rating", "detail_url"
    ]

    driver = init_driver()
    all_games = []

    try:
        print("\n[1단계] 순위 목록 수집")
        for page_num in range(1, TOTAL_PAGES + 1):
            print(f"  페이지 {page_num}/{TOTAL_PAGES} 수집 중...")
            games = get_rank_list_from_page(driver, page_num)
            all_games.extend(games)
            time.sleep(PAGE_PAUSE)

        print(f"\n  총 {len(all_games)}개 게임 링크 수집 완료")

        seen_urls = set()
        unique_games = []
        for g in all_games:
            if g["detail_url"] not in seen_urls:
                seen_urls.add(g["detail_url"])
                unique_games.append(g)
        print(f"  중복 제거 후: {len(unique_games)}개")
        # unique_games = unique_games[:2]  # 테스트

        print("\n[2단계] 상세 페이지 수집 (10개마다 중간 저장)")
        batch_games = []

        for i, game in enumerate(unique_games):
            # SAVE_INTERVAL마다 저장 후 driver 재시작
            if i > 0 and i % SAVE_INTERVAL == 0:
                save_games(batch_games, cols_order)
                print(f"  [중간 저장] {i}번째 게임까지 저장 완료")
                batch_games = []
                driver = restart_driver(driver)
                print(f"  driver 재시작")

            print(f"  ({i+1}/{len(unique_games)}) {game.get('title', '?')} 수집 중...")
            try:
                detail = get_detail_info(driver, game)
                batch_games.append(detail)

                # 리뷰 수집 비활성화
                # game_id = extract_game_id(game["detail_url"])
                # if game_id:
                #     reviews = get_reviews(driver, game_id)
                #     ...

            except Exception as e:
                print(f"    [오류] {e}")
                save_games(batch_games, cols_order)
                batch_games = []
                batch_games.append(game)
                driver = restart_driver(driver)
                print(f"    -> driver 재시작, 다음 게임으로 이동")

            time.sleep(DETAIL_PAUSE)

        save_games(batch_games, cols_order)
        print("\n[완료] boardlife_games.csv 저장 완료")

    finally:
        try:
            driver.quit()
        except Exception:
            pass
        print("\n크롤러 종료")


if __name__ == "__main__":
    main()