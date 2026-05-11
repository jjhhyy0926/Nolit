"""
머미나우 크롤러 v11 - 리뷰 전량 수집 버전
=========================================
컬럼:
  goods_no, name, category,
  description, author, publisher,
  rating, difficulty, players, play_time,
  total_rating, total_difficulty,
  review_count_normal, review_count_spoiler,
  review_text, review_author, review_date, review_rating, review_difficulty,
  release_date, image_url, url
"""

import csv
import json
import re
import time
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup


# ============================================================
# 설정
# ============================================================

DELAY_LIST      = 1.5
DELAY_DETAIL    = 1.5
DELAY_REVIEW    = 0.3
MAX_RETRIES     = 3
REQUEST_TIMEOUT = 15

OUTPUT_CSV  = "murmynow.csv"
OUTPUT_JSON = "murmynow.json"

HEADERS_PC = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.murmynow.com/",
}

HEADERS_MOBILE = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13; SM-S918B) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Mobile Safari/537.36"
    ),
    "Referer": "https://m.murmynow.com/",
}

CATEGORIES = {
    "001001": "개인운영",
    "001002": "플랫폼",
    "001003": "보드게임",
    "001004": "테마공연",
}

FIELDNAMES = [
    "goods_no", "name", "category",
    "description", "author", "publisher",
    "rating", "difficulty", "players", "play_time",
    "total_rating", "total_difficulty",
    "review_count_normal", "review_count_spoiler",
    "review_text", "review_author", "review_date",
    "review_rating", "review_difficulty",
    "release_date", "image_url", "url",
]


# ============================================================
# 유틸
# ============================================================

# 별점 width% → 점수 변환 (100%=5점)
def width_to_rating(style: str) -> str:
    m = re.search(r"width\s*:\s*(\d+)%", style)
    if m:
        return str(round(int(m.group(1)) / 20, 1))
    return ""


# ============================================================
# 로깅
# ============================================================

class Log:

    @staticmethod
    def header(t):
        print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")

    @staticmethod
    def step(m):
        print(f"\n  → {m}")

    @staticmethod
    def progress(c, t, l):
        print(f"  [{c:>3}/{t}] {l}")

    @staticmethod
    def result(m):
        print(f"\n  ✅ {m}")

    @staticmethod
    def warn(m):
        print(f"\n  ⚠️  {m}")

    @staticmethod
    def error(m):
        print(f"\n  ❌ {m}")


# ============================================================
# HTTP
# ============================================================

class HttpClient:

    def __init__(self):
        self.session = requests.Session()

    def get(self, url, headers=None, params=None):

        for attempt in range(1, MAX_RETRIES + 1):

            try:
                resp = self.session.get(
                    url,
                    headers=headers or HEADERS_MOBILE,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                resp.raise_for_status()

                return resp

            except Exception as e:

                if attempt < MAX_RETRIES:
                    time.sleep(attempt * 2)

                else:
                    Log.error(f"요청 실패: {url} — {e}")

        return None


# ============================================================
# 데이터 모델
# ============================================================

@dataclass
class ThemeItem:
    goods_no: str
    name: str
    category: str


@dataclass
class ThemeDetail:
    description: str = ""
    author: str = ""
    publisher: str = ""
    release_date: str = ""

    rating: str = ""
    difficulty: str = ""
    players: str = ""
    play_time: str = ""

    total_rating: str = ""
    total_difficulty: str = ""

    review_count_normal: str = ""
    review_count_spoiler: str = ""

    image_url: str = ""


@dataclass
class Review:
    review_text: str = ""
    review_author: str = ""
    review_date: str = ""

    review_rating: str = ""
    review_difficulty: str = ""


# ============================================================
# 크롤러
# ============================================================

class MurmynowCrawler:

    def __init__(self):
        self.client = HttpClient()

    # ----------------------------------------------------------
    # Step 1: 테마 목록
    # ----------------------------------------------------------

    def collect_theme_list(self):

        Log.header("Step 1 — 테마 목록 수집")

        all_themes = []
        seen = set()

        for cate_cd, cate_name in CATEGORIES.items():

            Log.step(f"카테고리: {cate_name}")

            page = 1

            while True:

                url = (
                    f"https://www.murmynow.com/goods/goods_list.php"
                    f"?cateCd={cate_cd}&page={page}"
                )

                resp = self.client.get(url, HEADERS_PC)

                if not resp:
                    break

                soup = BeautifulSoup(resp.text, "html.parser")

                count_before = len(all_themes)

                for link in soup.select('a[href*="goods_view.php?goodsNo="]'):

                    match = re.search(
                        r"goodsNo=(\d+)",
                        link.get("href", "")
                    )

                    if not match:
                        continue

                    goods_no = match.group(1)

                    if goods_no in seen:
                        continue

                    name_tag = link.select_one("strong, b")

                    name = (
                        name_tag.get_text(strip=True)
                        if name_tag
                        else link.get_text(strip=True)[:80]
                    )

                    if not name:
                        continue

                    seen.add(goods_no)

                    all_themes.append(
                        ThemeItem(goods_no, name, cate_name)
                    )

                added = len(all_themes) - count_before

                print(f"     p{page}: +{added}개 (누적 {len(all_themes)}개)")

                has_next = soup.select_one(
                    f'a[href*="page={page + 1}"]'
                )

                if added == 0 or not has_next:
                    break

                page += 1

                time.sleep(DELAY_LIST)

        Log.result(f"목록 {len(all_themes)}개 수집 완료")

        return all_themes

    # ----------------------------------------------------------
    # Step 2: 상세 정보
    # ----------------------------------------------------------

    def parse_detail(self, goods_no):

        url = (
            f"https://m.murmynow.com/goods/goods_view.php"
            f"?goodsNo={goods_no}"
        )

        resp = self.client.get(url, HEADERS_MOBILE)

        if not resp:
            return ThemeDetail()

        soup = BeautifulSoup(resp.text, "html.parser")

        d = ThemeDetail()

        # OG 이미지
        og = soup.find("meta", property="og:image")

        if og:
            d.image_url = og.get("content", "")

        # 테마 소개 / 작가 / 퍼블리셔
        sub_info = soup.select_one(".detail_sub_info")

        if sub_info:

            for dl in sub_info.select("dl"):

                dt = dl.select_one("dt")
                dd = dl.select_one("dd")

                if not dt or not dd:
                    continue

                key = dt.get_text(strip=True)
                val = dd.get_text(" ", strip=True)

                if "테마 소개" in key:
                    d.description = val

                elif "작가 정보" in key:
                    d.author = val

                elif "퍼블리셔 정보" in key:
                    d.publisher = val

        # 평점 / 난이도 / 인원 / 소요시간
        theme_info = soup.select_one(".theme_info")

        if theme_info:

            for box in theme_info.select(".item_box"):

                label = box.select_one("span")
                value = box.select_one("p")

                if not label or not value:
                    continue

                k = label.get_text(strip=True)
                v = value.get_text(strip=True)

                if k == "평점":
                    d.rating = v

                elif k == "난이도":
                    d.difficulty = v

                elif k == "인원":
                    d.players = v

                elif k == "소요시간":
                    d.play_time = v

        # 총 평점 / 총 난이도
        avg_box = soup.select_one(".theme_avg_score_box")

        if avg_box:

            rbox = avg_box.select_one(".item_goodsPt")

            if rbox:
                s = rbox.select_one(".item_text strong")

                if s:
                    d.total_rating = (
                        s.get_text(strip=True)
                        .split("/")[0]
                        .strip()
                    )

            dbox = avg_box.select_one(".item_difficulty")

            if dbox:
                s = dbox.select_one(".item_text strong")

                if s:
                    d.total_difficulty = (
                        s.get_text(strip=True)
                        .split("/")[0]
                        .strip()
                    )

        # 리뷰 수
        r_cnt = soup.select_one("li#detailReview .itemnum")

        if r_cnt:
            d.review_count_normal = r_cnt.get_text(strip=True)

        s_cnt = soup.select_one("li#spoilerReview .itemnum")

        if s_cnt:
            d.review_count_spoiler = s_cnt.get_text(strip=True)

        # 출시일
        info_text = soup.select_one(".js_goods_detail_infotext")

        if info_text:

            for dl in info_text.select("dl"):

                for dt, dd in zip(dl.select("dt"), dl.select("dd")):

                    if "출시일" in dt.get_text(strip=True):
                        d.release_date = dd.get_text(strip=True)

        return d

    # ----------------------------------------------------------
    # Step 3: 리뷰 (전량 수집 — 페이지네이션)
    # ----------------------------------------------------------

    def _parse_review_items(self, soup):
        """한 페이지의 리뷰 li 요소들을 파싱하여 Review 리스트로 반환"""

        reviews = []

        for item in soup.select("li.review_list_li"):

            # 별점
            rating = ""
            rating_span = item.select_one(".rating span[style]")
            if rating_span:
                rating = width_to_rating(
                    rating_span.get("style", "")
                )

            # 작성일
            date = ""
            date_el = item.select_one(".reg_date_box span")
            if date_el:
                date = date_el.get_text(strip=True)

            # 닉네임
            author = ""
            author_el = item.select_one(".member_name")
            if author_el:
                author = author_el.get_text(strip=True)

            # 본문
            text = ""
            text_el = item.select_one(".cont_box a")
            if text_el:
                text = text_el.get_text(strip=True)

            # 난이도
            difficulty = ""
            diff_el = item.select_one(".wg_icon_difficulty")
            if diff_el:
                difficulty = diff_el.get_text(strip=True)

            if text:
                reviews.append(
                    Review(
                        review_text=text,
                        review_author=author,
                        review_date=date,
                        review_rating=rating,
                        review_difficulty=difficulty,
                    )
                )

        return reviews

    def collect_reviews(self, goods_no):
        """테마의 모든 리뷰를 전량 수집

        머미나우(고도몰 커스텀 스킨) 리뷰 로드 방식:
        - 상세 페이지 방문으로 세션/쿠키 확보 필수 (없으면 403)
        - goods_board_list.php?bdId=goodsreview&goodsNo=...&gboard=y&page=N
        - 페이지당 리뷰 5개
        - 응답 HTML 내 button.detail_more_btn의 data-next-page로 다음 페이지 판별
        - data-next-page=0 또는 버튼 없음 → 종료

        주의: parse_detail이 먼저 호출되어 세션이 확보된 상태여야 함
        """

        all_reviews = []
        page = 1
        MAX_PAGES = 100  # 무한루프 안전장치

        while page <= MAX_PAGES:

            params = {
                "bdId": "goodsreview",
                "goodsNo": goods_no,
                "gboard": "y",
                "page": page,
            }

            resp = self.client.get(
                "https://m.murmynow.com/goods/goods_board_list.php",
                HEADERS_MOBILE,
                params=params,
            )

            if not resp or not resp.text.strip():
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            page_reviews = self._parse_review_items(soup)

            # 이 페이지에서 파싱된 리뷰가 없으면 종료
            if not page_reviews:
                break

            all_reviews.extend(page_reviews)

            # data-next-page로 다음 페이지 판별
            more_btn = soup.select_one("button.detail_more_btn")

            if not more_btn:
                break

            next_page = more_btn.get("data-next-page", "0")

            if next_page == "0" or not next_page:
                break

            page = int(next_page)
            time.sleep(DELAY_REVIEW)

        return all_reviews

    # ----------------------------------------------------------
    # 실행
    # ----------------------------------------------------------

    def run(self):

        themes = self.collect_theme_list()

        if not themes:
            Log.error("테마 없음.")
            return

        Log.header(
            f"Step 2-3 — 상세 + 리뷰 ({len(themes)}개)"
        )

        all_rows = []
        review_total = 0

        for i, theme in enumerate(themes, 1):

            Log.progress(i, len(themes), theme.name)

            detail = self.parse_detail(theme.goods_no)

            reviews = self.collect_reviews(theme.goods_no)

            review_total += len(reviews)

            if reviews:
                print(f"        ↳ 리뷰 {len(reviews)}건 수집")

            base = {
                **asdict(theme),
                **asdict(detail),
                "url": (
                    "https://www.murmynow.com/goods/goods_view.php"
                    f"?goodsNo={theme.goods_no}"
                ),
            }

            if not reviews:

                all_rows.append({
                    **base,
                    **asdict(Review())
                })

            else:

                for rv in reviews:

                    all_rows.append({
                        **base,
                        **asdict(rv)
                    })

            time.sleep(DELAY_DETAIL)

        self._save(all_rows)

        Log.header("수집 완료 요약")

        print(f"  테마 수     : {len(themes)}")
        print(f"  리뷰 수     : {review_total}")
        print(f"  총 행 수    : {len(all_rows)}")
        print(f"  CSV         : {OUTPUT_CSV}")
        print(f"  JSON        : {OUTPUT_JSON}")

    # ----------------------------------------------------------
    # 저장
    # ----------------------------------------------------------

    @staticmethod
    def _save(rows):

        if not rows:
            Log.warn("저장할 데이터 없음.")
            return

        keys = FIELDNAMES + [
            k for k in rows[0]
            if k not in FIELDNAMES
        ]

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=keys
            )

            writer.writeheader()
            writer.writerows(rows)

        with open(
            OUTPUT_JSON,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                rows,
                f,
                ensure_ascii=False,
                indent=2
            )

        Log.result(
            f"저장 완료 → {OUTPUT_CSV} / {OUTPUT_JSON}"
        )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    crawler = MurmynowCrawler()

    crawler.run()