"""
boardlife.co.kr 크롤러
- 대상: https://boardlife.co.kr/rank/all/{page}  (page 18~34)
- 추출: 게임 메타(19개 컬럼) / 리뷰
- 저장: output/boardlife_games.csv, output/boardlife_reviews.csv

설치:
    pip install -r requirements.txt
    python -m playwright install chromium

실행:
    python dasol_crawl_boardlife.py                       # 18~34 페이지 전체
    python dasol_crawl_boardlife.py --start 18 --end 18   # 특정 범위
    python dasol_crawl_boardlife.py --no-reviews          # 게임 메타만 수집
    python dasol_crawl_boardlife.py --max-review-pages 1  # 리뷰 1페이지만 (테스트)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page as PlaywrightPage

# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
BASE_URL    = "https://boardlife.co.kr"
RANK_URL    = BASE_URL + "/rank/all/{page}"
GAME_URL    = BASE_URL + "/game/{game_id}"
CREDITS_URL = BASE_URL + "/game/{game_id}/credits"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

ROOT            = Path(__file__).resolve().parent
OUTPUT_DIR      = ROOT / "output"
CHECKPOINT_FILE = OUTPUT_DIR / "_checkpoint.json"

GAMES_CSV   = OUTPUT_DIR / "boardlife_games.csv"
REVIEWS_CSV = OUTPUT_DIR / "boardlife_reviews.csv"

GAME_COLUMNS = [
    "rank", "title", "title_eng", "players", "recommended_players", "playing_time",
    "age", "weight", "designer", "artist", "description", "type",
    "category", "mechanism", "image", "avg_rating",
    "category_rank",
]

REVIEW_COLUMNS = [
    "rank", "title", "game_id", "rating", "reviewer",
    "date", "review_text", "review_page",
]

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("boardlife")


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------
@dataclass
class GameSummary:
    rank: int | None
    title: str | None
    title_eng: str | None
    game_id: str | None
    detail_url: str | None
    image: str | None = None
    avg_rating: str | None = None
    num_rating: str | None = None


@dataclass
class GameDetail:
    players: str | None = None
    recommended_players: str | None = None
    playing_time: str | None = None
    age: str | None = None
    weight: str | None = None
    designer: str | None = None
    artist: str | None = None
    description: str | None = None
    type: str | None = None
    category: str | None = None
    mechanism: str | None = None
    avg_rating: str | None = None
    image: str | None = None
    category_rank: str | None = None  # 예: "전체 1위 | 전략 1위 | 파티 3위"


@dataclass
class Review:
    rank: int | None
    title: str | None
    game_id: str | None
    rating: str | None
    reviewer: str | None
    date: str | None
    review_text: str | None
    review_page: int | None


# ---------------------------------------------------------------------------
# HTTP 페치 (재시도 + 랜덤 딜레이)
# ---------------------------------------------------------------------------
class Fetcher:
    def __init__(self, min_delay: float = 1.0, max_delay: float = 2.0,
                 max_retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.min_delay   = min_delay
        self.max_delay   = max_delay
        self.max_retries = max_retries

    def get(self, url: str) -> str | None:
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(random.uniform(self.min_delay, self.max_delay))
                resp = self.session.get(url, timeout=20)
                if resp.status_code == 200:
                    return resp.text
                log.warning(f"GET {url} -> {resp.status_code} (try {attempt})")
            except requests.RequestException as e:
                log.warning(f"GET {url} 실패: {e} (try {attempt})")
            time.sleep(min(2 ** attempt, 10))
        log.error(f"포기: {url}")
        return None


# ---------------------------------------------------------------------------
# 파서
# ---------------------------------------------------------------------------
def _text(node) -> str:
    return node.get_text(" ", strip=True) if node else ""


def parse_rank_page(html: str, page: int) -> list[GameSummary]:
    soup = BeautifulSoup(html, "html.parser")
    games: list[GameSummary] = []

    for row in soup.select("div.rank-row:not(.top)"):
        # game_id
        row_id = row.get("id", "")
        m = re.search(r"check-list-(\d+)", row_id)
        if not m:
            continue
        game_id = m.group(1)

        # 순위
        rank_el = row.select_one("div.rank [class*='digits']")
        rank_value = int(rank_el.get_text(strip=True)) if rank_el else None

        # 제목
        title_el = row.select_one("a.title")
        title = title_el.get_text(strip=True) if title_el else None
        if not title:
            continue

        # 영어 제목
        eng_el = row.select_one("div.bullet.eng")
        title_eng = eng_el.get_text(strip=True) if eng_el else None

        # 이미지
        img = row.select_one("div.thumb img")
        image = img.get("src") or img.get("data-src") if img else None

        # 평점
        rate_el = row.select_one("a.game-rate")
        avg_rating = rate_el.get_text(strip=True) if rate_el else None

        # 평점 없거나 "-"면 제외
        if not avg_rating or avg_rating == "-":
            continue

        games.append(GameSummary(
            rank=rank_value, title=title, title_eng=title_eng,
            game_id=game_id, detail_url=GAME_URL.format(game_id=game_id),
            image=image, avg_rating=avg_rating, num_rating=None,
        ))

    log.info(f"  page {page}: {len(games)} 게임 발견")
    return games


DETAIL_LABEL_MAP: dict[str, str] = {
    "인원": "players", "최소~최대 플레이 인원": "players",
    "추천 인원": "recommended_players", "BGG 추천 인원": "recommended_players",
    "플레이 시간": "playing_time", "플레이 소요 시간": "playing_time",
    "최소 연령": "age", "권장 연령": "age", "사용 연령": "age",
    "복잡도": "weight", "난이도": "weight",
    "디자이너": "designer", "아티스트": "artist",
    "타입": "type", "카테고리": "category",
    "메카닉": "mechanism", "메커니즘": "mechanism",
}


def parse_game_detail(html: str) -> GameDetail:
    soup = BeautifulSoup(html, "html.parser")
    detail = GameDetail()

    play_boxes = soup.select("div.game-play-info a.game-play-box")
    if play_boxes:
        for box in play_boxes:
            data = _text(box.find("div", class_="data") or box)
            unit = _text(box.find("div", class_="unit") or None)
            if "명" in unit and not detail.players:
                detail.players = data + unit.split()[0]
            elif "분" in unit and not detail.playing_time:
                detail.playing_time = data + unit.split()[0]
            elif "세" in unit and not detail.age:
                detail.age = data + unit.split()[0]
        for info_box in soup.select("a.game-info-box, div.game-info-box"):
            title_node = info_box.find("div", class_="title")
            data_node  = info_box.find("div", class_="data")
            if not title_node or not data_node:
                continue
            label = _text(title_node)
            col = DETAIL_LABEL_MAP.get(label)
            if col and not getattr(detail, col):
                setattr(detail, col, _text(data_node))

    for row in soup.select("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        label = _text(cells[0]).rstrip(":：").strip()
        value = _text(cells[1])
        col = DETAIL_LABEL_MAP.get(label)
        if col and not getattr(detail, col):
            setattr(detail, col, value)

    for dt in soup.select("dt"):
        label = _text(dt).rstrip(":：").strip()
        col = DETAIL_LABEL_MAP.get(label)
        if not col:
            continue
        dd = dt.find_next_sibling("dd")
        if dd and not getattr(detail, col):
            setattr(detail, col, _text(dd))

    desc_node = None
    for header in soup.find_all(string=re.compile(r"(소개|설명|Description)", re.I)):
        parent = header.parent
        if parent is None:
            continue
        nxt = parent.find_next(["p", "div"])
        if nxt and len(_text(nxt)) > 30:
            desc_node = nxt
            break
    if desc_node is not None:
        detail.description = _text(desc_node)
    else:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            detail.description = meta["content"].strip()

    rec = soup.find("div", class_="recommend-player")
    if rec:
        rec_text = _text(rec)
        if not detail.recommended_players:
            detail.recommended_players = rec_text.strip("() ")
        if detail.players:
            detail.players = detail.players.replace(rec_text, "").strip()

    # 카테고리별 순위: "전략 순위", "워게임 순위" 등 전체 제외
    ranks = []
    for dt in soup.select("dt.page-title"):
        label = dt.get_text(strip=True)
        if "순위" not in label or label in ("순위",):
            continue
        rank_span = dt.find_next("span", class_="game-rank-num")
        if rank_span:
            inner = rank_span.find("span")
            val = inner.get_text(strip=True).replace(",", "") if inner else None
            if val and val.isdigit():
                ranks.append(f"{label} {val}위")
    detail.category_rank = " | ".join(ranks) if ranks else None

    # 상세 페이지 고화질 이미지
    for sel in ["div.main-img img", "img.main-img", "div.photo img", "div.thumb img"]:
        img_el = soup.select_one(sel)
        if img_el:
            src = img_el.get("src") or img_el.get("data-src")
            if src and "/svg/" not in src:
                detail.image = urljoin(BASE_URL, src)
                break

    # weight
    weight_span = soup.find("span", id="game-weight")
    if weight_span:
        w = weight_span.get_text(strip=True)
        if w and w != "-":
            detail.weight = w

    # avg_rating
    rate_svg = soup.find("div", class_="main-play-rate-svg")
    if rate_svg:
        rating_span = rate_svg.find_next_sibling("span", class_="main-color")
        if rating_span:
            detail.avg_rating = rating_span.get_text(strip=True)

    return detail


CREDITS_LABEL_MAP: dict[str, str] = {
    "아트웍 작가": "artist", "카테고리": "type",
    "테마": "category", "진행방식": "mechanism", "디자이너": "designer",
}


def parse_credits_page(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, str] = {}
    for wrapper in soup.select("div.title-wrapper.credit"):
        label_node = wrapper.select_one("div.title")
        if not label_node:
            continue
        count_div = label_node.find("div", class_="count")
        if count_div:
            count_div.extract()
        label = label_node.get_text(strip=True)
        col = CREDITS_LABEL_MAP.get(label)
        if not col:
            continue
        values = [
            _text(a)
            for a in wrapper.select("div.credits-row a")
            if _text(a)
        ]
        if values:
            result[col] = ", ".join(values)
    return result


# ---------------------------------------------------------------------------
# Playwright 리뷰 수집
# ---------------------------------------------------------------------------
def parse_reviews_from_html(html: str, *, rank, title, game_id: str,
                             page: int) -> list[Review]:
    soup = BeautifulSoup(html, "html.parser")
    reviews = []

    for row in soup.select("div.rate-row"):
        point = row.select_one(".rate-point")
        rating = point.get_text(strip=True) if point else None

        if not rating:
            continue

        title_rate = row.select_one(".rate-comment .title-rate")
        comment    = row.select_one(".rate-comment")
        review_text = None
        if title_rate:
            review_text = title_rate.get_text(strip=True) or None
        elif comment:
            review_text = comment.get_text(strip=True) or None

        if review_text:
            review_text = review_text.replace("\n", " ").replace("\r", " ").strip()

        nick = row.select_one(".nick") or row.select_one(".toNick")
        reviewer = nick.get_text(strip=True) if nick else None

        date_el = row.select_one(".date")
        date = date_el.get_text(strip=True) if date_el else None

        reviews.append(Review(
            rank=rank, title=title, game_id=game_id,
            rating=rating, reviewer=reviewer, date=date,
            review_text=review_text, review_page=page,
        ))

    return reviews


def fetch_reviews_playwright(pw_page: PlaywrightPage, *, rank, title,
                              game_id: str, page: int) -> tuple[list[Review], bool]:
    url = f"{BASE_URL}/game/{game_id}/rate"
    if page > 1:
        url += f"?page={page}"

    pw_page.goto(url, wait_until="domcontentloaded", timeout=30000)
    time.sleep(2)

    try:
        pw_page.wait_for_selector("div.rate-row", timeout=8000)
    except Exception:
        return [], False

    html = pw_page.content()
    reviews = parse_reviews_from_html(html, rank=rank, title=title,
                                      game_id=game_id, page=page)

    soup = BeautifulSoup(html, "html.parser")
    paging = soup.select("a.paging-btn")
    has_next = any(
        str(page + 1) in (a.get("href") or "") or
        a.get_text(strip=True) in ["다음", ">", "»"]
        for a in paging
    )

    return reviews, has_next


# ---------------------------------------------------------------------------
# CSV 저장 도우미
# ---------------------------------------------------------------------------
class CsvWriter:
    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.columns = columns
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        self.fp = path.open("a", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.fp, fieldnames=columns,
                                     extrasaction="ignore")
        if new_file:
            self.writer.writeheader()
            self.fp.flush()

    def write_row(self, row: dict):
        self.writer.writerow(row)
        self.fp.flush()

    def close(self):
        self.fp.close()


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------
def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        try:
            return set(json.loads(CHECKPOINT_FILE.read_text("utf-8")))
        except json.JSONDecodeError:
            return set()
    return set()


def save_checkpoint(done: set[str]):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False, indent=2), "utf-8"
    )


# ---------------------------------------------------------------------------
# 메인 흐름
# ---------------------------------------------------------------------------
def crawl(start_page: int, end_page: int, *, use_cache: bool,
          max_review_pages: int | None, skip_reviews: bool = False):

    fetcher = Fetcher()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    games_writer   = CsvWriter(GAMES_CSV, GAME_COLUMNS)
    reviews_writer = CsvWriter(REVIEWS_CSV, REVIEW_COLUMNS)
    done_ids = load_checkpoint()
    log.info(f"이미 처리한 게임: {len(done_ids)} 건 (체크포인트)")

    # 1) 순위 페이지 수집 (Playwright)
    all_games: list[GameSummary] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], locale="ko-KR")
        pw = context.new_page()
        for page in range(start_page, end_page + 1):
            url = RANK_URL.format(page=page)
            log.info(f"[순위] page {page}: {url}")
            pw.goto(url, wait_until="domcontentloaded", timeout=30000)
            pw.wait_for_timeout(3000)
            all_games.extend(parse_rank_page(pw.content(), page))
        browser.close()

    log.info(f"총 {len(all_games)} 개 게임 발견 (page {start_page}~{end_page})")

    # 2) 게임별 상세 + 리뷰 수집
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=DEFAULT_HEADERS["User-Agent"],
            locale="ko-KR",
        )
        pw_page = context.new_page()

        for idx, gs in enumerate(all_games, 1):
            if not gs.game_id:
                continue
            if gs.game_id in done_ids:
                log.info(f"[{idx}/{len(all_games)}] {gs.title} — skip")
                continue

            log.info(f"[{idx}/{len(all_games)}] {gs.title} (id={gs.game_id})")

            # 2-1) 상세 + credits
            detail_html = fetcher.get(gs.detail_url) if gs.detail_url else None
            detail = parse_game_detail(detail_html) if detail_html else GameDetail()

            credits_html = fetcher.get(CREDITS_URL.format(game_id=gs.game_id))
            if credits_html:
                for col, val in parse_credits_page(credits_html).items():
                    if not getattr(detail, col):
                        setattr(detail, col, val)

            # 2-2) 리뷰 수집 (--no-reviews 시 건너뜀)
            review_count = 0
            if not skip_reviews:
                page = 1
                while True:
                    reviews, has_next = fetch_reviews_playwright(
                        pw_page,
                        rank=gs.rank, title=gs.title,
                        game_id=gs.game_id, page=page,
                    )
                    if not reviews:
                        break

                    for rv in reviews:
                        reviews_writer.write_row(asdict(rv))
                    review_count += len(reviews)
                    log.info(f"    리뷰 page {page}: +{len(reviews)} (누적 {review_count})")

                    if max_review_pages is not None and page >= max_review_pages:
                        break
                    if not has_next:
                        break
                    page += 1

            # 2-3) 게임 메타 저장 (평점·난이도 없으면 제외, 리뷰 수집 시 1개 이하면 제외)
            avg = detail.avg_rating or gs.avg_rating
            if not avg or not detail.weight:
                log.info(f"    평점/난이도 없음 → skip")
                done_ids.add(gs.game_id)
                save_checkpoint(done_ids)
                continue
            if not skip_reviews and review_count <= 1:
                log.info(f"    리뷰 부족 → skip")
                done_ids.add(gs.game_id)
                save_checkpoint(done_ids)
                continue

            gs.num_rating = review_count
            row = {**asdict(gs), **asdict(detail)}
            games_writer.write_row(row)

            done_ids.add(gs.game_id)
            save_checkpoint(done_ids)

        browser.close()

    games_writer.close()
    reviews_writer.close()
    log.info("크롤링 완료")
    log.info(f"  게임 CSV : {GAMES_CSV}")
    log.info(f"  리뷰 CSV : {REVIEWS_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="보드라이프 크롤러")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=34)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-review-pages", type=int, default=None)
    parser.add_argument("--no-reviews", action="store_true",
                        help="리뷰 수집 건너뜀 (게임 메타만 수집)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    crawl(
        start_page=args.start,
        end_page=args.end,
        use_cache=not args.no_cache,
        max_review_pages=args.max_review_pages,
        skip_reviews=args.no_reviews,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())