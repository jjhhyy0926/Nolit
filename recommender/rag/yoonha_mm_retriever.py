"""
yoonha_mm_retriever.py
머더미스터리로그 BM25 + FAISS RRF 하이브리드 검색

====================================================================
[역할]
    머더미스터리로그 데이터를 대상으로 BM25(키워드)와 FAISS(의미 벡터)
    두 가지 검색을 병행한 뒤 RRF(Reciprocal Rank Fusion)로 점수를
    통합하여 최종 순위를 산출한다.

[구조 — boardgame_retriever와 동일한 패턴]
    1. 모듈 로드 시점에 FAISS 인덱스 + 메타 JSON + BM25 코퍼스 초기화
    2. hard_filter()  → 인원/시간/지역 하드 필터
    3. _bm25_search() → 키워드 기반 검색
    4. _dense_search() → FAISS 의미 벡터 검색
    5. _rrf_fuse()     → 두 결과를 RRF 로 융합 + 메타데이터 가중치
    6. retrieve()      → 1~5 를 조합한 공개 인터페이스

[설계 문서와의 대응]
    - 기획서 §5: BM25 + FAISS 하이브리드 검색
    - 필터 문서 §머더미스터리: min/max_players, play_time, area 하드 필터
    - 필터 문서 §머더미스터리: rating 가중치, difficulty(이산형) 가중치
    - 필터 문서 §공통 주의사항 §2: None 값은 0점이 아님

[boardgame_retriever와의 차이점]
    - 단일 소스(murdermysterylog)만 사용 → 멀티소스 RRF 불필요
    - RRF 스케일링 계수가 1000 (boardgame은 3000)
    - metadata_weight가 rating × 2 로 단순 (boardgame은 카테고리·메커니즘·인원 등 복합)
    - 필터 문서 §공통 주의사항 §5: 지역 필터는 빠방만 가능 → 여기서는 향후 확장용으로 area 필터 존재하나 머더미스터리로그에 지역 데이터 없음

[개선 포인트]
    1. 머미나우(murmynow) 소스 미통합 — 기획서에는 머미나우+머더로그 두 소스 사용 명시
    2. difficulty 가중치 미구현 — 필터 문서에 이산형(1~4) 가중치 설계 있음
    3. scene_category 하드 필터 미구현 — 필터 문서에 "배우 참여형" 등 유형 필터 명시
    4. RRF k=60은 boardgame과 동일 — 단일 소스라 k 조정 여지 있음

변경사항:
  - _metadata_weight(): reviews 텍스트 기반 키워드 보너스 추가
    (리뷰가 meta에 포함되어 있으므로 별도 파일 로드 불필요)
  - 2차 가중치: rank, rating, 리뷰 텍스트 길이(데이터 풍부도) 반영
  - rating 스케일: 0~5 (기획서 명시)
"""

import json
import re
import math
import faiss
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

# -------------------------
# 데이터 로드
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

_index = faiss.read_index(str(DATA_DIR / "faiss_murdermysterylog.index"))
with open(DATA_DIR / "faiss_murdermysterylog_meta.json", "r", encoding="utf-8") as f:
    all_items = json.load(f)

for item in all_items:
    item["source"] = "murdermysterylog"
    if "name" in item and "title" not in item:
        item["title"] = item["name"]

print(f"[mm_retriever] 머더미스터리: {len(all_items)}개 (dim={_index.d})")

# -------------------------
# BM25 준비
# -------------------------
def _make_searchable_text(item: dict) -> str:
    parts = [str(item.get("title", item.get("name", "")))]
    if item.get("description"):
        parts.append(str(item["description"])[:500])
    if item.get("시리즈"):
        parts.append(str(item["시리즈"]))
    if item.get("제작"):
        parts.append(str(item["제작"]))
    if item.get("reviews"):
        # reviews 텍스트 전체를 BM25 corpus에 포함 (|| 구분자 그대로 활용)
        parts.append(str(item["reviews"]))
    for tag in item.get("emotion_tags", []):
        parts.append(str(tag))
    return " ".join(parts)


_corpus = [_make_searchable_text(s) for s in all_items]
_tokenized_corpus = [c.lower().split() for c in _corpus]
_bm25 = BM25Okapi(_tokenized_corpus)
print(f"[mm_retriever] BM25 corpus 준비 완료: {len(_corpus)}개")



# -------------------------
# 메타데이터 정규화 유틸
# -------------------------
def _as_number(value, default=None):
    """문자열/숫자 메타데이터를 float로 안전 변환. None은 0으로 보지 않는다."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned or cleaned.lower() in {"none", "null", "nan", "?", "-"}:
            return default
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return default
    return default


def _as_int(value, default=None):
    num = _as_number(value, default=None)
    return int(num) if num is not None else default


def _contains(value, target: str) -> bool:
    if not target:
        return False
    if value is None:
        return False
    return str(target).lower() in str(value).lower()

# -------------------------
# reviews 텍스트 유틸
# -------------------------
def _get_reviews_text(item: dict) -> str:
    """reviews 필드에서 텍스트 추출. || 구분자로 연결된 문자열 그대로 반환."""
    reviews = item.get("reviews", "")
    return str(reviews) if reviews else ""


def _count_keyword(text: str, keywords: list[str]) -> int:
    """텍스트에서 키워드 출현 횟수 합산."""
    text_lower = text.lower()
    return sum(text_lower.count(kw.lower()) for kw in keywords)


# -------------------------
# 하드 필터
# -------------------------
def hard_filter(item: dict, query_filter: dict) -> bool:
    """
    조건 불만족 아이템 제거. True = 통과.
    문서 기준: 인원, 시간, scene_category만 하드 필터로 적용한다.
    """
    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        max_p = _as_int(item.get("max_players"), default=999)
        min_p = _as_int(item.get("min_players"), default=0)
        if players > max_p or players < min_p:
            return False

    max_time = _as_int(query_filter.get("max_time"), default=None)
    if max_time is not None:
        # 머미나우는 min_time/max_time 범위형, 머더로그는 play_time 단일값
        item_time = _as_int(item.get("max_time"), default=None)
        if item_time is None:
            item_time = _as_int(item.get("play_time"), default=None)
        if item_time is not None and item_time > 0 and item_time > max_time:
            return False

    scene_category = query_filter.get("scene_category")
    if scene_category:
        item_scene = item.get("scene_category") or item.get("유형") or item.get("type") or ""
        if item_scene and not _contains(item_scene, scene_category):
            return False

    # 지역 필드는 머더미스터리로그에는 보통 없으므로, 데이터가 있을 때만 적용한다.
    if query_filter.get("area"):
        item_area = item.get("area") or item.get("지역") or ""
        if item_area and not _contains(item_area, query_filter["area"]):
            return False

    if query_filter.get("location"):
        item_location = item.get("location") or item.get("장소") or ""
        if item_location and not _contains(item_location, query_filter["location"]):
            return False

    return True

# -------------------------
# 메타데이터 가중치
# -------------------------
def _metadata_weight(item: dict, query_filter: dict) -> float:
    """
    메타데이터 기반 가중치 계산.
    - rating: 0~5 스케일, 높을수록 우선
    - difficulty: 머미나우 1/2/3/4 이산형 기준. 낮을수록 쉬움, 높을수록 어려움
    - None은 0점이 아니라 데이터 없음으로 처리
    """
    score = 0.0

    # 1. rating (0~5 스케일)
    rating = _as_number(item.get("rating"), default=None)
    if rating is not None and rating > 0:
        score += min(rating / 5.0, 1.0) * 12.0

    # 2. difficulty preference
    difficulty = _as_number(item.get("difficulty"), default=None)
    pref = query_filter.get("difficulty_pref")
    if pref and difficulty is not None:
        # 1=쉬움, 2=보통, 3=어려움, 4=매우 어려움
        if pref == "light":
            score += max(0.0, (3.0 - difficulty) / 2.0) * 6.0
        elif pref == "medium":
            score += max(0.0, 1.0 - abs(difficulty - 2.0) / 2.0) * 5.0
        elif pref == "heavy":
            score += max(0.0, (difficulty - 2.0) / 2.0) * 6.0

    # 3. horror preference: 데이터가 있을 때만 반영
    horror = _as_number(item.get("horror"), default=None)
    horror_pref = query_filter.get("horror_pref")
    if horror_pref and horror is not None:
        # 0~5, 높을수록 공포 강함
        if horror_pref == "low":
            score += max(0.0, (5.0 - horror) / 5.0) * 5.0
        elif horror_pref == "medium":
            score += max(0.0, 1.0 - abs(horror - 2.5) / 2.5) * 4.0
        elif horror_pref == "high":
            score += max(0.0, horror / 5.0) * 5.0

    # 4. scene_category soft boost
    if query_filter.get("scene_category"):
        item_scene = item.get("scene_category") or item.get("유형") or item.get("type") or ""
        if _contains(item_scene, query_filter["scene_category"]):
            score += 5.0

    # 5. reviews 텍스트 키워드 보너스
    reviews_text = _get_reviews_text(item)
    if reviews_text:
        emotion_tags = query_filter.get("emotion_tags", [])
        for tag in emotion_tags:
            hits = _count_keyword(reviews_text, [tag])
            if hits > 0:
                score += min(hits * 0.5, 3.0)

        positive_kw = ["재밌", "재미있", "명작", "수작", "추천", "최고", "몰입", "만족"]
        negative_kw = ["별로", "실망", "지루", "노잼", "비추", "최악", "아쉬"]
        pos_hits = _count_keyword(reviews_text, positive_kw)
        neg_hits = _count_keyword(reviews_text, negative_kw)
        sentiment_score = min(pos_hits * 0.03 - neg_hits * 0.05, 3.0)
        score += max(sentiment_score, -3.0)

        review_count = reviews_text.count("||") + 1
        score += min(math.log1p(review_count) / math.log1p(50), 1.0) * 2.0

    return score

# -------------------------
# 검색 내부 함수
# -------------------------
def _bm25_search(query_text: str, query_filter: dict, topk: int = 200) -> dict:
    tokens = query_text.lower().split()
    scores = _bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1]
    results = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(all_items): continue
        item = all_items[idx]
        if not hard_filter(item, query_filter): continue
        title = item.get("title", item.get("name", ""))
        if title not in results:
            rank += 1
            results[title] = {"item": item, "rank": rank, "bm25_score": float(scores[idx])}
            if rank >= topk: break
    return results


def _dense_search(query_vector: np.ndarray, query_filter: dict, topk: int = 200) -> dict:
    if query_vector.shape[1] != _index.d:
        raise ValueError(f"쿼리 벡터 dim 불일치: {query_vector.shape[1]} != {_index.d}")
    D, I = _index.search(query_vector, topk * 3)
    results = {}
    rank = 0
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(all_items): continue
        item = all_items[idx]
        if not hard_filter(item, query_filter): continue
        title = item.get("title", item.get("name", ""))
        if title not in results:
            rank += 1
            results[title] = {"item": item, "rank": rank, "l2_dist": float(D[0][i])}
            if rank >= topk: break
    return results


def _rrf_fuse(
    bm25_results: dict,
    dense_results: dict,
    query_filter: dict,
    topk: int,
    k: int = 60,
) -> list[dict]:
    all_titles = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []
    for title in all_titles:
        bm25_data  = bm25_results.get(title)
        dense_data = dense_results.get(title)
        bm25_rank  = bm25_data["rank"]  if bm25_data  else 999
        dense_rank = dense_data["rank"] if dense_data else 999
        if bm25_rank == 999 and dense_rank == 999:
            continue

        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)
        item = (bm25_data or dense_data)["item"]
        meta_score  = _metadata_weight(item, query_filter)
        total_score = rrf_score * 1000 + meta_score

        item_copy = item.copy()
        item_copy["rrf_score"]   = round(rrf_score, 6)
        item_copy["meta_score"]  = round(meta_score, 2)
        item_copy["total_score"] = round(total_score, 2)
        item_copy["bm25_rank"]   = bm25_rank
        item_copy["dense_rank"]  = dense_rank
        scored.append(item_copy)

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored[:topk]


# -------------------------
# 앵커 임베딩 유틸
# -------------------------
def get_embedding(titles: list[str]) -> np.ndarray:
    """
    타이틀 리스트 → 평균 임베딩 벡터 반환 (shape: (1, dim)).
    타이틀을 찾지 못하면 인덱스 0번 벡터 사용.
    """
    embeddings = []
    for title in titles:
        for i, s in enumerate(all_items):
            if s.get("title") == title or s.get("name") == title:
                embeddings.append(_index.reconstruct(i))
                break
    if embeddings:
        return np.mean(embeddings, axis=0).reshape(1, -1).astype(np.float32)
    return _index.reconstruct(0).reshape(1, -1).astype(np.float32)


# -------------------------
# 공개 인터페이스
# -------------------------
def retrieve(
    query_text: str,
    query_filter: dict,
    query_vector: np.ndarray,
    topk: int = 50,
) -> list[dict]:
    """RRF 하이브리드 검색 (BM25 + FAISS 융합)."""
    bm25_res  = _bm25_search(query_text, query_filter, topk=200)
    dense_res = _dense_search(query_vector, query_filter, topk=200)
    return _rrf_fuse(bm25_res, dense_res, query_filter, topk=topk)


def retrieve_bm25(
    query_text: str,
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """BM25 단독 검색."""
    results = _bm25_search(query_text, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]


def retrieve_dense(
    query_vector: np.ndarray,
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """Dense 단독 검색."""
    results = _dense_search(query_vector, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]


def retrieve_vanilla(
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """Vanilla 검색 — 필터 통과 후 평점 내림차순."""
    results = [item.copy() for item in all_items if hard_filter(item, query_filter)]
    results.sort(key=lambda x: x.get("rating") or 0, reverse=True)
    return results[:topk]