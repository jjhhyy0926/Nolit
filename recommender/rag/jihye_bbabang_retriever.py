"""
recommender/rag/jihye_bbabang_retriever.py

빠방 방탈출 BM25 + FAISS RRF 하이브리드 검색

====================================================================
[ 역할 ]
    빠방 데이터를 대상으로 BM25(키워드)와 FAISS(의미 벡터) 검색을 병행한 뒤
    RRF(Reciprocal Rank Fusion)로 점수를 통합하고 메타데이터 가중치를 적용한다.
    yoonha_hybrid_retriever.py에서 category="escape"일 때 라우팅된다.

[ 데이터 소스 ]
    faiss_bbabang_stats.index / _metadata.json  : 테마별 통계 (843개)
        fields: id, source, title, store_name, area, location, playing_time,
                max_players, price, difficulty, horror, activity, satisfaction,
                puzzle, story, interior, production, avg_headcount, description, address
    faiss_bbabang_reviews.index / _metadata.json: 리뷰 청크 (11769개)
        fields: id, source, title, store_name, chunk_index, document

[ 검색 흐름 ]
    1. BM25: stats(title+description+address) + 리뷰 집계 텍스트로 검색
    2. FAISS: stats 인덱스(768dim × 843)로 의미 유사도 검색
    3. RRF: 두 랭킹을 통합 (k=60)
    4. hard_filter: area, location, max_players, price, playing_time 제거
    5. metadata_weight: satisfaction, horror, difficulty, puzzle, story, interior, production

[ 하드 필터 (기획서 §방탈출) ]
    area       : 시/도 (강원, 경기, 인천)
    location   : 시/군/구 (원주시, 강릉시, 수원시 ...)
    max_players: 쿼리 인원이 최대 인원 초과 시 제외 (빠방은 min 없음)
    price      : 예산 초과 시 제외
    playing_time: 플레이 시간 초과 시 제외

[ 가중치 (기획서 §방탈출) ]
    horror     : 0~5, 쿼리 방향에 따라 (low → 낮을수록 우선 / high → 높을수록)
    difficulty : 0~5 연속형, 쿼리 방향에 따라
    satisfaction: 0~5, 높을수록 우선 (전체 만족도)
    puzzle     : 0~5, prefer_puzzle=True 시 높을수록 우선
    story      : 0~5, prefer_story=True 시 높을수록 우선
    interior   : 0~5 (실측 max≈6.0), prefer_interior=True 시 높을수록
    production : 0~5 (실측 max≈6.5), prefer_production=True 시 높을수록

[ RRF 스케일 ]
    total_score = rrf_score × 1000 + metadata_weight
    (단일 소스라 스케일 1000 사용 — boardgame은 3000, mm도 1000)
====================================================================
"""

import json
import math
import re
import faiss
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "04_vectorstore"

_stats_index = faiss.read_index(str(DATA_DIR / "faiss_bbabang_stats.index"))
with open(DATA_DIR / "faiss_bbabang_stats_metadata.json", "r", encoding="utf-8") as f:
    all_stats = json.load(f)

with open(DATA_DIR / "faiss_bbabang_reviews_metadata.json", "r", encoding="utf-8") as f:
    _reviews_raw = json.load(f)

for item in all_stats:
    item["source"] = "bbabang"

print(f"[bbabang_retriever] 테마 통계: {len(all_stats)}개 (dim={_stats_index.d})")

# ──────────────────────────────────────────
# 리뷰 텍스트 집계: (title, store_name) → 합산 문서
# ──────────────────────────────────────────
_review_agg: dict[tuple, str] = {}
for r in _reviews_raw:
    key = (r.get("title", "").strip(), r.get("store_name", "").strip())
    doc = r.get("document", "")
    if doc:
        _review_agg.setdefault(key, [])
        _review_agg[key].append(doc)

for key in _review_agg:
    _review_agg[key] = " ".join(_review_agg[key])[:2000]  # 토큰 제한

print(f"[bbabang_retriever] 리뷰 집계 완료: {len(_review_agg)}개 테마")

# ──────────────────────────────────────────
# BM25 코퍼스 구성 (stats 기준, 리뷰 텍스트 보강)
# ──────────────────────────────────────────
def _make_searchable_text(item: dict) -> str:
    parts = [
        str(item.get("title", "")),
        str(item.get("store_name", "")),
    ]
    if item.get("area"):
        parts.append(str(item["area"]))
    if item.get("location"):
        parts.append(str(item["location"]))
    if item.get("address"):
        parts.append(str(item["address"]))
    if item.get("description"):
        parts.append(str(item["description"])[:300])

    key = (item.get("title", "").strip(), item.get("store_name", "").strip())
    review_text = _review_agg.get(key, "")
    if review_text:
        parts.append(review_text[:1000])

    return " ".join(p for p in parts if p)


_corpus = [_make_searchable_text(s) for s in all_stats]
_tokenized_corpus = [c.split() for c in _corpus]
_bm25 = BM25Okapi(_tokenized_corpus)
print(f"[bbabang_retriever] BM25 corpus 준비 완료: {len(_corpus)}개")


# ──────────────────────────────────────────
# 메타데이터 정규화 유틸
# ──────────────────────────────────────────
def _as_number(value, default=None):
    """문자열/숫자 메타데이터를 float로 안전 변환. None은 그대로 None."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            if math.isnan(value) or math.isinf(value):
                return default
        except TypeError:
            pass
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned or cleaned.lower() in {"none", "null", "nan", "na", "n/a", "?", "-"}:
            return default
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if m:
            try:
                num = float(m.group(0))
                if math.isnan(num) or math.isinf(num):
                    return default
                return num
            except ValueError:
                return default
    return default


def _as_int(value, default=None):
    num = _as_number(value, default=None)
    if num is None:
        return default
    try:
        if math.isnan(num) or math.isinf(num):
            return default
    except TypeError:
        return default
    return int(num)


# ──────────────────────────────────────────
# 하드 필터
# ──────────────────────────────────────────
def hard_filter(item: dict, query_filter: dict) -> bool:
    """
    조건 불만족 아이템 제거. True = 통과.

    [기획서 §방탈출 하드 필터]
        area       : 시/도 불일치 시 제외
        location   : 시/군/구 불일치 시 제외
        max_players: 쿼리 인원이 아이템 최대 인원 초과 시 제외
                     (빠방은 min_players 없으므로 max_players만 체크)
        price      : 예산 초과 시 제외
        playing_time: 플레이 시간 초과 시 제외
    """
    # 지역 (시/도)
    if query_filter.get("area"):
        item_area = item.get("area") or ""
        if item_area != query_filter["area"]:
            return False

    # 세부 지역 (시/군/구)
    if query_filter.get("location"):
        item_loc = item.get("location") or ""
        if item_loc != query_filter["location"]:
            return False

    # 인원 — 빠방은 max_players만 존재
    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        max_p = _as_int(item.get("max_players"), default=None)
        if max_p is None:
            return False  # 인원 데이터 없으면 제외
        if players > max_p:
            return False

    # 가격
    price_limit = _as_int(query_filter.get("price"), default=None)
    if price_limit is not None:
        item_price = _as_int(item.get("price"), default=None)
        if item_price is not None and item_price > price_limit:
            return False

    # 플레이 시간
    time_limit = _as_int(query_filter.get("playing_time"), default=None)
    if time_limit is not None:
        item_time = _as_int(item.get("playing_time"), default=None)
        if item_time is not None and item_time > time_limit:
            return False

    return True


# ──────────────────────────────────────────
# 메타데이터 가중치
# ──────────────────────────────────────────
def _metadata_weight(item: dict, query_filter: dict) -> float:
    """
    메타데이터 기반 2차 가중치 계산.

    [방탈출 가중치]
        horror     : 0~5, 높을수록 공포 강함. 0이면 공포 없음
        difficulty : 0~5 연속형 (빠방 기준), 높을수록 어려움
        satisfaction: 0~5, 높을수록 우선
        puzzle     : 0~5, prefer_puzzle 시 높을수록
        story      : 0~5, prefer_story 시 높을수록
        interior   : 0~5 (실측 max≈6.0), prefer_interior 시 높을수록
        production : 0~5 (실측 max≈6.5), prefer_production 시 높을수록

    [None 값 처리]
        None은 0점이 아니라 데이터 없음. 가중치 계산에서 제외.
    """
    score = 0.0

    # 1. satisfaction: 전체 만족도, 높을수록 우선 (0~5)
    satisfaction = _as_number(item.get("satisfaction"), default=None)
    if satisfaction is not None:
        score += (satisfaction / 5.0) * 10.0

    # 2. horror: 쿼리 방향에 따라 (0~5, 높을수록 공포 강함)
    horror = _as_number(item.get("horror"), default=None)
    horror_pref = query_filter.get("horror_pref")
    if horror_pref and horror is not None:
        if horror_pref == "low":
            # 안 무서운 거 원함 → horror 낮을수록 가점
            score += max(0.0, (5.0 - horror) / 5.0) * 8.0
        elif horror_pref == "high":
            # 무서운 거 원함 → horror 높을수록 가점
            score += (horror / 5.0) * 8.0
        # medium: 특별 가중치 없음 (중립)

    # 3. difficulty: 쿼리 방향에 따라 (0~5 연속형)
    difficulty = _as_number(item.get("difficulty"), default=None)
    difficulty_pref = query_filter.get("difficulty_pref")
    if difficulty_pref and difficulty is not None:
        if difficulty_pref in ("light", "low"):
            score += max(0.0, (5.0 - difficulty) / 5.0) * 6.0
        elif difficulty_pref in ("heavy", "high"):
            score += (difficulty / 5.0) * 6.0
        elif difficulty_pref == "medium":
            score += max(0.0, 1.0 - abs(difficulty - 2.5) / 2.5) * 5.0

    # 4. puzzle: prefer_puzzle 시 높을수록 (0~5)
    if query_filter.get("prefer_puzzle"):
        puzzle = _as_number(item.get("puzzle"), default=None)
        if puzzle is not None:
            score += (puzzle / 5.0) * 5.0

    # 5. story: prefer_story 시 높을수록 (0~5)
    if query_filter.get("prefer_story"):
        story = _as_number(item.get("story"), default=None)
        if story is not None:
            score += (story / 5.0) * 5.0

    # 6. interior: prefer_interior 시 높을수록 (실측 max≈6.0 → 정규화)
    if query_filter.get("prefer_interior"):
        interior = _as_number(item.get("interior"), default=None)
        if interior is not None:
            score += min(interior / 6.0, 1.0) * 5.0

    # 7. production: prefer_production 시 높을수록 (실측 max≈6.5 → 정규화)
    if query_filter.get("prefer_production"):
        production = _as_number(item.get("production"), default=None)
        if production is not None:
            score += min(production / 6.5, 1.0) * 5.0

    return score


# ──────────────────────────────────────────
# 검색 내부 함수
# ──────────────────────────────────────────
def _bm25_search(query_text: str, query_filter: dict, topk: int = 200) -> dict:
    """BM25 키워드 검색 → hard_filter 적용 → 상위 topk 반환."""
    tokens = query_text.split()
    scores = _bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1]

    results = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(all_stats):
            continue
        item = all_stats[idx]
        if not hard_filter(item, query_filter):
            continue
        key = f"{item.get('title', '')}::{item.get('store_name', '')}"
        if key not in results:
            rank += 1
            results[key] = {
                "item": item,
                "rank": rank,
                "bm25_score": float(scores[idx]),
                "idx": int(idx),
            }
            if rank >= topk:
                break
    return results


def _dense_search(query_vector: np.ndarray, query_filter: dict, topk: int = 200) -> dict:
    """FAISS stats 인덱스 dense 검색 → hard_filter 적용 → 상위 topk 반환."""
    if query_vector.shape[1] != _stats_index.d:
        raise ValueError(
            f"쿼리 벡터 dim 불일치: {query_vector.shape[1]} != {_stats_index.d}"
        )
    D, I = _stats_index.search(query_vector, topk * 3)

    results = {}
    rank = 0
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(all_stats):
            continue
        item = all_stats[idx]
        if not hard_filter(item, query_filter):
            continue
        key = f"{item.get('title', '')}::{item.get('store_name', '')}"
        if key not in results:
            rank += 1
            results[key] = {
                "item": item,
                "rank": rank,
                "l2_dist": float(D[0][i]),
                "idx": int(idx),
            }
            if rank >= topk:
                break
    return results


def _rrf_fuse(
    bm25_results: dict,
    dense_results: dict,
    query_filter: dict,
    topk: int,
    k: int = 60,
) -> list[dict]:
    """
    RRF 융합 + 메타데이터 가중치 적용.

    total_score = rrf_score × 1000 + metadata_weight
    """
    all_keys = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []

    for key in all_keys:
        bm25_data = bm25_results.get(key)
        dense_data = dense_results.get(key)
        bm25_rank = bm25_data["rank"] if bm25_data else 999
        dense_rank = dense_data["rank"] if dense_data else 999
        if bm25_rank == 999 and dense_rank == 999:
            continue

        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)

        # dense 없을 때 BM25 단독 패널티
        if dense_rank == 999:
            rrf_score *= 0.7

        item = (bm25_data or dense_data)["item"]
        meta_score = _metadata_weight(item, query_filter)
        total_score = rrf_score * 1000 + meta_score

        item_copy = item.copy()
        item_copy["rrf_score"] = round(rrf_score, 6)
        item_copy["meta_score"] = round(meta_score, 2)
        item_copy["total_score"] = round(total_score, 2)
        item_copy["bm25_rank"] = bm25_rank
        item_copy["dense_rank"] = dense_rank
        scored.append(item_copy)

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored[:topk]


# ──────────────────────────────────────────
# 앵커 임베딩 유틸
# ──────────────────────────────────────────
def get_embedding(titles: list[str]) -> np.ndarray:
    """
    타이틀 리스트 → 평균 임베딩 벡터 반환 (shape: (1, dim=768)).
    타이틀을 찾지 못하면 stats 인덱스 0번 벡터 사용.
    """
    embeddings = []
    for title in titles:
        title_stripped = title.strip()
        for i, s in enumerate(all_stats):
            if s.get("title", "").strip() == title_stripped:
                embeddings.append(_stats_index.reconstruct(i))
                break
    if embeddings:
        return np.mean(embeddings, axis=0).reshape(1, -1).astype(np.float32)
    return _stats_index.reconstruct(0).reshape(1, -1).astype(np.float32)


# ──────────────────────────────────────────
# 공개 인터페이스
# ──────────────────────────────────────────
def retrieve(
    query_text: str,
    query_filter: dict,
    query_vector: np.ndarray,
    topk: int = 50,
) -> list[dict]:
    """
    BM25 + FAISS RRF 하이브리드 검색.

    Args:
        query_text  : BM25 키워드 쿼리 (query_transformer 생성)
        query_filter: 하드 필터 조건 dict
                      keys: area, location, players, price, playing_time,
                            horror_pref, difficulty_pref,
                            prefer_puzzle, prefer_story, prefer_interior, prefer_production
        query_vector: FAISS 검색용 벡터 (1, 768)
        topk        : 반환 최대 개수

    Returns:
        total_score 내림차순 정렬된 아이템 리스트
    """
    bm25_res = _bm25_search(query_text, query_filter, topk=200)
    dense_res = _dense_search(query_vector, query_filter, topk=200)
    return _rrf_fuse(bm25_res, dense_res, query_filter, topk=topk)


def retrieve_bm25(
    query_text: str,
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """BM25 단독 검색 (평가/디버깅용)."""
    results = _bm25_search(query_text, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]


def retrieve_dense(
    query_vector: np.ndarray,
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """Dense 단독 검색 (평가/디버깅용)."""
    results = _dense_search(query_vector, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]


def retrieve_vanilla(
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """Vanilla 검색 — 필터 통과 후 satisfaction 내림차순 (FAISS 장애 fallback)."""
    results = [item.copy() for item in all_stats if hard_filter(item, query_filter)]
    results.sort(key=lambda x: x.get("satisfaction") or 0, reverse=True)
    return results[:topk]
