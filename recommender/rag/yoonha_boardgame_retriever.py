"""
recommender/rag/yoonha_boardgame_retriever.py

[ 역할 ]
  BGG + 보드라이프 통합 BM25 + FAISS RRF 하이브리드 검색

[ 검색 흐름 ]
  1. BM25 키워드 검색 → 키워드 매칭 기반 랭킹
  2. FAISS dense 검색 → 의미 유사도 기반 랭킹
  3. RRF(Reciprocal Rank Fusion) → 두 랭킹을 결합
  4. 메타데이터 가중치 → 평점/카테고리/인원 등 보정
  5. 최종 total_score 계산 후 정렬

[ RRF 공식 ]
  rrf_score = 1/(k + bm25_rank) + 1/(k + dense_rank)
  - k=60 (기본값): 높은 랭크에 집중하면서도 중간 랭크도 반영
  - BM25에만 있는 아이템: rrf_score × 0.7 (dense 미등장 패널티)

[ 최종 점수 ]
  total_score = rrf_score × 3000 + metadata_weight
  - 3000 곱하기: rrf_score(0.001~0.03)를 metadata_weight(0~30)와 스케일 맞춤

[ 데이터 소스별 특이사항 ]
  - BGG: 10점 만점, playing_time 단일값, weight(복잡도) 0~5
  - 보드라이프: 5점 만점, min_time/max_time 범위형, best_players 존재
  - 보드라이프 가중치 1.5배 (한국 서비스이므로 한국 데이터 우선)

변경사항:
  - faiss_bgg_reviews_meta.json 로드 → title별 review_count / review_avg / review_stdev 집계
  - faiss_boardlife_reviews_meta.json 로드 → 동일
  - _metadata_weight(): 리뷰 수 신뢰도 + 호불호(stdev) 반영
  - RRF 소스 가중치: boardlife × 1.5, bgg × 1.0 (기획서 권장값)
  - 2차 가중치: rank, avg_rating, review_count 순서로 반영
"""

import json
import math
import re
import statistics
import faiss
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

# -------------------------
# 데이터 로드
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

bgg_index = faiss.read_index(str(DATA_DIR / "faiss_bgg_stats.index"))
with open(DATA_DIR / "faiss_bgg_stats_meta.json", "r", encoding="utf-8") as f:
    bgg_stats = json.load(f)

bl_index = faiss.read_index(str(DATA_DIR / "faiss_boardlife_stats.index"))
with open(DATA_DIR / "faiss_boardlife_stats_meta.json", "r", encoding="utf-8") as f:
    bl_stats = json.load(f)

for item in bgg_stats:
    item["source"] = "bgg"
for item in bl_stats:
    item["source"] = "boardlife"

all_items = bgg_stats + bl_stats

print(f"[boardgame_retriever] BGG: {len(bgg_stats)}개 (dim={bgg_index.d}), 보드라이프: {len(bl_stats)}개 (dim={bl_index.d})")

# -------------------------
# reviews meta 로드 및 집계
# -------------------------

def _build_review_map(data: list[dict]) -> dict[str, dict]:
    """
    reviews meta → title별 통계 dict 빌드.
    {
      "Brass: Birmingham": {
          "review_count": 4557,
          "review_avg":   8.344,
          "review_stdev": 1.810,
      }, ...
    }
    """
    raw: dict[str, list[float]] = {}
    for x in data:
        r = x.get("rating")
        if r is None:
            continue
        raw.setdefault(x["title"], []).append(float(r))

    result = {}
    for title, ratings in raw.items():
        result[title] = {
            "review_count": len(ratings),
            "review_avg":   round(sum(ratings) / len(ratings), 3),
            "review_stdev": round(statistics.stdev(ratings), 3) if len(ratings) > 1 else 0.0,
        }
    return result


with open(DATA_DIR / "faiss_bgg_reviews_meta.json", "r", encoding="utf-8") as f:
    _bgg_review_map = _build_review_map(json.load(f))

with open(DATA_DIR / "faiss_boardlife_reviews_meta.json", "r", encoding="utf-8") as f:
    _bl_review_map = _build_review_map(json.load(f))

print(f"[boardgame_retriever] BGG 리뷰 집계: {len(_bgg_review_map)}개 게임")
print(f"[boardgame_retriever] 보드라이프 리뷰 집계: {len(_bl_review_map)}개 게임")


def _get_review_data(item: dict) -> dict | None:
    """아이템 소스에 맞는 review 집계 데이터 반환."""
    title = item.get("title", "")
    if item.get("source") == "bgg":
        return _bgg_review_map.get(title)
    else:
        # boardlife: title(한글) 기준으로 먼저 탐색, 없으면 title_eng로 bgg map 보조 탐색
        return _bl_review_map.get(title) or _bgg_review_map.get(item.get("title_eng", ""))


# -------------------------
# BM25 준비
# -------------------------
CATEGORY_KO = {
    "Strategy": "전략", "Economic": "경제", "Party": "파티", "War": "전쟁",
    "Family": "가족", "Abstract": "추상", "Thematic": "테마", "Adventure": "어드벤처",
    "Fantasy": "판타지", "Horror": "공포", "Science Fiction": "SF",
    "Deduction": "추리", "Negotiation": "협상", "Cooperative": "협력",
    "Card Game": "카드게임", "Dice": "주사위", "Puzzle": "퍼즐",
}
MECHANISM_KO = {
    "Worker Placement": "일꾼배치", "Deck Building": "덱빌딩", "Engine Building": "엔진빌딩",
    "Area Control": "지역장악", "Cooperative Game": "협력", "Auction": "경매",
    "Market": "시장", "Hand Management": "패관리", "Tile Placement": "타일배치",
    "Route Building": "루트빌딩", "Push Your Luck": "운빨", "Voting": "투표",
    "Drafting": "드래프팅", "Roll and Write": "롤앤라이트",
}
BM25_STOPWORDS = {
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "a","an","the","of","in","at","to","for","and","or","is","it",
    "city","cities","world","age","game","games","edition","player","players"
}


def _translate_tags(tags_raw, mapping):
    if isinstance(tags_raw, list):
        tags = tags_raw
    elif isinstance(tags_raw, str) and tags_raw:
        tags = tags_raw.split("|")
    else:
        return []
    result = list(tags)
    for tag in tags:
        ko = mapping.get(tag.strip())
        if ko:
            result.append(ko)
    return result


def _make_searchable_text(item):
    parts = [str(item.get("title", ""))]
    if item.get("title_eng"):
        parts.append(str(item["title_eng"]))
    parts.extend(_translate_tags(item.get("category", ""), CATEGORY_KO))
    parts.extend(_translate_tags(item.get("mechanism", ""), MECHANISM_KO))
    t = item.get("type", "")
    if isinstance(t, list):
        parts.extend(t)
    elif isinstance(t, str) and t:
        parts.append(t)
    des = item.get("designer", "")
    if isinstance(des, list):
        parts.extend(des)
    elif isinstance(des, str) and des:
        parts.extend(des.split("|"))
    for tag in item.get("emotion_tags", []):
        parts.append(str(tag))
    return " ".join([str(p) for p in parts if p])


_corpus = [_make_searchable_text(s) for s in all_items]
_tokenized_corpus = [c.lower().split() for c in _corpus]
_bm25 = BM25Okapi(_tokenized_corpus)
print(f"[boardgame_retriever] BM25 corpus 준비 완료: {len(_corpus)}개")



# -------------------------
# 메타데이터 정규화 유틸
# -------------------------
def _as_number(value, default=None):
    """문자열/숫자 메타데이터를 float로 안전 변환. None/NaN/inf는 default로 처리."""
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


def _split_tags(value) -> list[str]:
    """list 또는 |/, 구분 문자열을 태그 리스트로 변환."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [t.strip() for t in re.split(r"[|,/]", value) if t.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _contains_tag(value, target: str) -> bool:
    if not target:
        return False
    target_lower = target.lower()
    tags = _split_tags(value)
    expanded = []
    expanded.extend(tags)
    expanded.extend(_translate_tags(tags, CATEGORY_KO))
    expanded.extend(_translate_tags(tags, MECHANISM_KO))
    return any(target_lower in t.lower() or t.lower() in target_lower for t in expanded)


def _parse_player_values(value) -> set[int]:
    """
    recommended_players / best_players가
    int, float, list, 문자열, NaN 어떤 형태여도 안전하게 set[int]로 변환.
    NaN/None은 데이터 없음으로 처리한다.
    """
    values: set[int] = set()

    if value is None:
        return values

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return values
        values.add(int(value))
        return values

    if isinstance(value, int):
        values.add(value)
        return values

    if isinstance(value, list):
        for v in value:
            values |= _parse_player_values(v)
        return values

    if isinstance(value, dict):
        for v in value.values():
            values |= _parse_player_values(v)
        return values

    text = str(value).strip()

    if not text or text.lower() in {"none", "null", "nan", "na", "n/a", "-", "?"}:
        return values

    for m in re.findall(r"\d+", text):
        try:
            values.add(int(m))
        except ValueError:
            continue

    return values


def _normalized_rating(item: dict) -> float | None:
    """소스별 평점 스케일을 0~1로 정규화. None은 그대로 None."""
    rating = _as_number(item.get("avg_rating"), default=None)
    if rating is None or rating <= 0:
        return None
    source = item.get("source")
    if source == "bgg":
        return min(rating / 10.0, 1.0)
    if source == "boardlife":
        return min(rating / 5.0, 1.0)
    return None


def _source_weight(item: dict, query_filter: dict) -> float:
    """
    소스 가중치. 한국어/한국 유저 반응 중심이면 boardlife 우선, 글로벌이면 동등.
    기본값은 서비스 성격상 boardlife 1.5, bgg 1.0.
    """
    pref = query_filter.get("source_pref") or query_filter.get("source_preference") or "korean"
    if pref in {"global", "bgg"}:
        return 1.0
    if item.get("source") == "boardlife":
        return 1.5
    return 1.0

# -------------------------
# 하드 필터
# -------------------------
def hard_filter(item: dict, query_filter: dict) -> bool:
    """
    조건 불만족 아이템 제거. True = 통과.

    하드 필터는 문서 기준으로 인원/시간처럼 명백히 조건을 벗어나는 항목만 제거한다.
    weight는 기본적으로 가중치 영역이며, strict_weight_filter=True일 때만 제외한다.
    """
    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        max_p = _as_int(item.get("max_players"), default=999)
        min_p = _as_int(item.get("min_players"), default=0)
        if players > max_p or players < min_p:
            return False

    max_time = _as_int(query_filter.get("playing_time"), default=None)
    if max_time is not None:
        if item.get("source") == "boardlife":
            item_time = _as_int(item.get("max_time"), default=None)
        else:
            item_time = _as_int(item.get("playing_time"), default=None)
        if item_time is not None and item_time > 0 and item_time > max_time:
            return False

    # 선택적 엄격 필터. 일반 추천에서는 weight를 하드 제외하지 않고 rerank에만 사용한다.
    if query_filter.get("strict_weight_filter"):
        w = _as_number(item.get("weight"), default=None)
        if w is not None:
            weight_max = _as_number(query_filter.get("weight_max"), default=None)
            if weight_max is not None and w > weight_max:
                return False
            if query_filter.get("weight_pref") == "heavy" and w < 3.5:
                return False

    return True

# -------------------------
# 메타데이터 가중치
# -------------------------
def _metadata_weight(item: dict, query_filter: dict) -> float:
    """
    메타데이터 기반 2차 가중치 계산.
    - 평점은 소스별 정규화 후 비교한다.
    - None은 0점이 아니라 데이터 없음으로 처리한다.
    - source 가중치는 RRF 단계에서 한 번만 적용한다.
    """
    score = 0.0
    source = item.get("source", "bgg")

    # 1. avg_rating: source-aware normalization + 리뷰 신뢰도/호불호 보정
    normalized = _normalized_rating(item)
    if normalized is not None:
        review_data = _get_review_data(item)
        if review_data:
            count = _as_number(review_data.get("review_count"), default=0) or 0
            stdev = _as_number(review_data.get("review_stdev"), default=0) or 0
            confidence_base = 500.0 if source == "bgg" else 50.0
            confidence = min(count / confidence_base, 1.0)
            stdev_penalty = min(stdev / 4.5, 1.0) * 0.2
            score += normalized * 15.0 * (0.6 + 0.4 * confidence) * (1.0 - stdev_penalty)
        else:
            score += normalized * 15.0 * 0.6

    # 2. category/mechanism: 쿼리 의도 매칭
    if query_filter.get("category") and _contains_tag(item.get("category"), query_filter["category"]):
        score += 10.0
    if query_filter.get("mechanism") and _contains_tag(item.get("mechanism"), query_filter["mechanism"]):
        score += 8.0

    # 3. 추천/베스트 인원 일치
    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        if source == "bgg":
            rec_values = _parse_player_values(item.get("recommended_players"))
        else:
            rec_values = _parse_player_values(item.get("best_players")) | _parse_player_values(item.get("recommend_players")) | _parse_player_values(item.get("recommended_players"))
        if players in rec_values:
            score += 6.0

    # 4. weight preference: 0~5, 높을수록 복잡
    pref = query_filter.get("weight_pref")
    w = _as_number(item.get("weight"), default=None)
    if pref and w is not None:
        if pref == "light":
            score += max(0.0, (3.0 - w) / 3.0) * 6.0
        elif pref == "medium":
            score += max(0.0, 1.0 - abs(w - 3.0) / 2.0) * 6.0
        elif pref == "heavy":
            score += max(0.0, (w - 2.5) / 2.5) * 6.0

    # 5. category_rank(Overall): 낮을수록 우수
    cr = item.get("category_rank")
    overall = None
    if isinstance(cr, dict):
        overall = _as_number(cr.get("Overall"), default=None)
    elif isinstance(cr, str):
        try:
            cr_dict = json.loads(cr.replace("'", '"'))
            overall = _as_number(cr_dict.get("Overall") or cr_dict.get("전략") or cr_dict.get("가족"), default=None)
        except Exception:
            overall = _as_number(cr, default=None)
    if overall is not None and overall > 0:
        score += max(0.0, 10.0 - math.log1p(overall) * 1.35)

    # 6. rank: 낮을수록 우수
    rank = _as_number(item.get("rank"), default=None)
    if rank is not None and rank > 0:
        score += max(0.0, 5.0 - math.log1p(rank) * 0.75)

    # 7. review_count: 리뷰 수 신뢰도 보정
    review_data = _get_review_data(item)
    if review_data:
        count = _as_number(review_data.get("review_count"), default=0) or 0
        base = 500.0 if source == "bgg" else 50.0
        score += min(math.log1p(count) / math.log1p(base), 1.0) * 3.0

    return score

# -------------------------
# 검색 내부 함수
# -------------------------
def _bm25_search(query_text: str, query_filter: dict, topk: int = 200) -> dict:
    tokens = [t for t in query_text.lower().split() if t not in BM25_STOPWORDS]
    scores = _bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1]
    results = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(all_items): continue
        item = all_items[idx]
        if not hard_filter(item, query_filter): continue
        key = f"{item['source']}::{item.get('title', '')}"
        if key not in results:
            rank += 1
            results[key] = {"item": item, "rank": rank, "bm25_score": float(scores[idx])}
            if rank >= topk: break
    return results


def _dense_search(query_vector: np.ndarray, query_filter: dict, topk: int = 200) -> dict:
    """BGG + 보드라이프 각각 검색 후 l2_dist 기준 합산 정렬."""
    results = {}
    for index, meta in [(bgg_index, bgg_stats), (bl_index, bl_stats)]:
        if query_vector.shape[1] != index.d:
            continue
        D, I = index.search(query_vector, topk * 3)
        for i, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(meta): continue
            item = meta[idx]
            if not hard_filter(item, query_filter): continue
            key = f"{item['source']}::{item.get('title', '')}"
            if key not in results:
                results[key] = {"item": item, "l2_dist": float(D[0][i])}

    # l2_dist 기준 재정렬 후 rank 부여
    sorted_items = sorted(results.items(), key=lambda x: x[1]["l2_dist"])
    ranked = {}
    for rank, (key, data) in enumerate(sorted_items[:topk], 1):
        ranked[key] = {**data, "rank": rank}
    return ranked


def _rrf_fuse(
    bm25_results: dict,
    dense_results: dict,
    query_filter: dict,
    topk: int,
    k: int = 60,
) -> list[dict]:
    """
    RRF 융합 + 메타 가중치 적용.

    소스 가중치 (기획서 권장):
      boardlife: RRF 점수 × 1.5
      bgg:       RRF 점수 × 1.0
    """
    all_keys = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []
    for key in all_keys:
        bm25_data  = bm25_results.get(key)
        dense_data = dense_results.get(key)
        bm25_rank  = bm25_data["rank"]  if bm25_data  else 999
        dense_rank = dense_data["rank"] if dense_data else 999
        if bm25_rank == 999 and dense_rank == 999:
            continue

        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)

        # dense 없을 때 BM25 단독 패널티
        if dense_rank == 999:
            rrf_score *= 0.7

        item = (bm25_data or dense_data)["item"]

        # 소스 가중치: 한국어/한국 유저 반응 중심이면 boardlife 우선
        source_weight = _source_weight(item, query_filter)
        rrf_weighted  = rrf_score * source_weight

        meta_score  = _metadata_weight(item, query_filter)
        total_score = rrf_weighted * 3000 + meta_score

        item_copy = item.copy()
        item_copy["rrf_score"]    = round(rrf_score, 6)
        item_copy["meta_score"]   = round(meta_score, 2)
        item_copy["total_score"]  = round(total_score, 2)
        item_copy["bm25_rank"]    = bm25_rank
        item_copy["dense_rank"]   = dense_rank
        item_copy["source_weight"] = source_weight
        scored.append(item_copy)

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored[:topk]


# -------------------------
# 앵커 임베딩 유틸
# -------------------------
def get_embedding(titles: list[str]) -> np.ndarray:
    """
    타이틀 리스트 → 평균 임베딩 벡터 반환 (shape: (1, dim)).
    타이틀을 찾지 못하면 bgg 인덱스 0번 벡터 사용.
    """
    embeddings = []
    for title in titles:
        for index, meta in [(bgg_index, bgg_stats), (bl_index, bl_stats)]:
            for i, s in enumerate(meta):
                if s.get("title") == title or s.get("title_eng") == title or s.get("name") == title:
                    embeddings.append(index.reconstruct(i))
                    break
    if embeddings:
        return np.mean(embeddings, axis=0).reshape(1, -1).astype(np.float32)
    return bgg_index.reconstruct(0).reshape(1, -1).astype(np.float32)


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
    results.sort(key=lambda x: x.get("avg_rating") or 0, reverse=True)
    return results[:topk]