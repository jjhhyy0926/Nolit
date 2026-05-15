"""
yoonha_eval_pipeline.py

Nolit 4단계 RAG 파이프라인 통합 평가 스크립트

평가 항목:
    1. query_transformer 변환 확인
    2. retriever 검색 결과 비교
       - RRF
       - BM25
       - Dense
       - Vanilla
    3. Title Ground Truth Precision@10
       - 미리 지정한 정답 제목이 top10에 포함되는지 확인
    4. Condition@10
       - top10 결과가 query의 hard condition을 만족하는지 확인
       - 보드게임: 인원 + 시간
       - 머더미스터리: 인원 + 시간 + scene_category
       - ground_truth가 없어도 계산 가능
    5. tag_filter 필터링 확인
    6. generator 생성 확인
    7. graph 파이프라인 E2E 확인
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


# =========================================================
# Import path 설정
# - python -m recommender.eval.yoonha_eval_pipeline
# - python recommender/eval/yoonha_eval_pipeline.py
# 둘 다 대응
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = PROJECT_ROOT / "recommender" / "rag"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RAG_DIR))


from recommender.rag.yoonha_query_transformer import transform as query_transform
from recommender.rag.yoonha_hybrid_retriever import (
    retrieve,
    retrieve_bm25,
    retrieve_dense,
    retrieve_vanilla,
    get_embedding,
)
from recommender.rag.yoonha_tag_filter import filter_and_score
from recommender.rag.yoonha_generator import generate_without_api
from recommender.yoonha_graph import run_pipeline


# =========================================================
# 평가 쿼리 정의
# =========================================================

BOARDGAME_QUERIES = [
    {
        "name": "4인 전략 보드게임 (무거운)",
        "user_text": "4명이서 할 전략 보드게임",
        "group": {
            "headcount": 4,
            "play_time": 120,
            "weight_pref": "heavy",
            "category": "Strategy",
            "horror_tolerance": 2,
            "relation": "friend",
        },
        "ground_truth": [
            "Brass: Birmingham",
            "브라스: 버밍엄",
            "Twilight Struggle",
        ],
    },
    {
        "name": "2인 가벼운 파티게임",
        "user_text": "2명이서 가볍게 할 게임",
        "group": {
            "headcount": 2,
            "play_time": 60,
            "weight_pref": "light",
            "category": "Party",
            "horror_tolerance": 2,
            "relation": "couple",
        },
        "ground_truth": [
            "Codenames",
            "Dixit",
            "코드네임",
        ],
    },
    {
        "name": "3인 협력 게임",
        "user_text": "3명이서 협력하는 보드게임",
        "group": {
            "headcount": 3,
            "play_time": 120,
            "weight_pref": "medium",
            "category": "Cooperative",
            "horror_tolerance": 2,
            "relation": "friend",
        },
        "ground_truth": [
            "Pandemic",
            "Spirit Island",
            "팬데믹",
            "정령섬",
        ],
    },
]


MURDER_QUERIES = [
    {
        "name": "6인 쉬운 입문 머더미스터리",
        "user_text": "6명이서 할 쉬운 머더미스터리",
        "group": {
            "headcount": 6,
            "play_time": 180,
            "difficulty_pref": "light",
            "horror_tolerance": 0,
            "relation": "first_meeting",
        },
        "ground_truth": [
            "구두룡 저택의 살인",
            "몇 번이고 푸른 달에 불을 붙였다",
        ],
    },
    {
        "name": "4인 머더미스터리",
        "user_text": "4명이서 할 머더미스터리",
        "group": {
            "headcount": 4,
            "play_time": 240,
            "horror_tolerance": 2,
            "relation": "friend",
        },
        # 정답셋이 없는 smoke test 케이스.
        # Precision@10은 N/A가 되지만 Condition@10은 계산된다.
        "ground_truth": [],
    },
    {
        "name": "8인 대규모 파티",
        "user_text": "8명이서 할 파티 머더미스터리",
        "group": {
            "headcount": 8,
            "play_time": 300,
            "horror_tolerance": 1,
            "relation": "friend",
        },
        "ground_truth": [
            "구두룡 저택의 살인",
        ],
    },
]


# =========================================================
# 기본 출력 유틸
# =========================================================

def print_header(text: str):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_subheader(text: str):
    print(f"\n--- {text} ---")


def print_items(items, max_show=5):
    for i, item in enumerate(items[:max_show], 1):
        title = item.get("title", item.get("name", "?"))
        score = (
            item.get("final_score")
            or item.get("total_score")
            or item.get("avg_rating")
            or item.get("rating")
            or "?"
        )
        source = item.get("source", "?")
        print(f"  {i}. {title} (점수: {score}, 소스: {source})")


# =========================================================
# 평가 유틸: Title Ground Truth Precision@K
# =========================================================

def precision_at_k(items, ground_truth, k=10):
    """
    Title Ground Truth Precision@K

    ground_truth에 정의된 정답 제목이 top-k 결과에 포함되는지 확인한다.
    ground_truth가 비어 있으면 None을 반환하고, 출력에서는 N/A로 표시한다.
    """
    if not ground_truth:
        return None

    pred_titles = [
        item.get("title", item.get("name", ""))
        for item in items[:k]
    ]

    hits = 0

    for gt in ground_truth:
        for title in pred_titles:
            if gt and gt in title:
                hits += 1
                break

    return hits / k


# =========================================================
# 평가 유틸: Condition@K
# =========================================================

def _num(value, default=None):
    """
    숫자/문자열/NaN 값을 안전하게 숫자로 변환한다.
    None/NaN은 default로 처리한다.
    """
    if value is None:
        return default

    if isinstance(value, (int, float)):
        try:
            if value != value:  # NaN
                return default
        except Exception:
            return default
        return float(value)

    try:
        import re

        text = str(value).strip().replace(",", "")

        if not text or text.lower() in {
            "none",
            "null",
            "nan",
            "na",
            "n/a",
            "-",
            "?",
        }:
            return default

        m = re.search(r"-?\d+(?:\.\d+)?", text)

        if not m:
            return default

        num = float(m.group(0))

        if num != num:  # NaN
            return default

        return num

    except Exception:
        return default


def _contains_any(value, targets):
    """
    문자열/list/dict 값 안에 target keyword가 포함되는지 확인한다.

    value가 None이면 데이터가 없는 것이므로 조건 불일치로 단정하지 않는다.
    즉 optional metadata는 없다고 해서 fail 처리하지 않는다.
    """
    if value is None:
        return True

    if not isinstance(targets, list):
        targets = [targets]

    if isinstance(value, dict):
        value_text = " ".join(str(v) for v in value.values())
    elif isinstance(value, list):
        value_text = " ".join(str(v) for v in value)
    else:
        value_text = str(value)

    value_text = value_text.lower()

    return any(
        str(t).lower() in value_text
        for t in targets
        if t
    )


def _players_pass(item, players):
    """
    인원 조건 만족 여부.

    players가 None이면 해당 조건은 평가하지 않는다.
    min_players가 없으면 0, max_players가 없으면 999로 처리한다.
    """
    if players is None:
        return True

    min_p = _num(item.get("min_players"), 0)
    max_p = _num(item.get("max_players"), 999)

    return min_p <= players <= max_p


def _time_pass(item, max_time, category):
    """
    시간 조건 만족 여부.

    max_time이 None이면 해당 조건은 평가하지 않는다.
    시간 데이터가 없는 경우에는 조건 불일치로 단정하지 않는다.
    """
    if max_time is None:
        return True

    if category == "boardgame":
        if item.get("source") == "boardlife":
            item_time = _num(item.get("max_time"), None)
        else:
            item_time = _num(item.get("playing_time"), None)

        if item_time is None:
            item_time = _num(item.get("max_time"), None)

        if item_time is None:
            item_time = _num(item.get("play_time"), None)

    else:
        item_time = _num(item.get("max_time"), None)

        if item_time is None:
            item_time = _num(item.get("play_time"), None)

        if item_time is None:
            item_time = _num(item.get("playing_time"), None)

    if item_time is None or item_time <= 0:
        return True

    return item_time <= max_time


def _boardgame_condition_pass(item, query_filter):
    """
    보드게임 Condition Pass 여부.

    Condition@10은 hard filter 조건만 평가한다.
    즉, 인원/시간처럼 반드시 만족해야 하는 조건만 본다.

    category, mechanism, weight_pref는 soft preference/reranking 요소이므로
    Condition@10의 fail 조건으로 사용하지 않는다.
    """
    players = query_filter.get("players")
    max_time = query_filter.get("playing_time")

    if not _players_pass(item, players):
        return False

    if not _time_pass(item, max_time, "boardgame"):
        return False

    return True


def _murder_condition_pass(item, query_filter):
    """
    머더미스터리 Condition Pass 여부.

    Condition@10은 hard filter 조건만 평가한다.
    즉, 인원/시간/명시적 scene_category만 본다.

    difficulty_pref, horror_pref는 soft preference/reranking 요소이므로
    Condition@10의 fail 조건으로 사용하지 않는다.
    """
    players = query_filter.get("players")
    max_time = query_filter.get("max_time")

    if not _players_pass(item, players):
        return False

    if not _time_pass(item, max_time, "murdermystery"):
        return False

    scene_category = query_filter.get("scene_category")
    if scene_category:
        scene_value = (
            item.get("scene_category")
            or item.get("type")
            or item.get("유형")
        )

        if scene_value is not None and not _contains_any(scene_value, scene_category):
            return False

    return True


def condition_pass_at_k(items, query_filter, category, k=10):
    """
    Condition@K

    top-k 결과 중 query의 hard condition을 만족하는 결과의 비율.
    ground_truth title이 아니라 조건 충족 여부만 본다.

    ground_truth가 비어 있어도 계산 가능하다.
    """
    topk = items[:k]

    if not topk:
        return 0.0

    passed = 0

    for item in topk:
        if category == "boardgame":
            ok = _boardgame_condition_pass(item, query_filter)
        else:
            ok = _murder_condition_pass(item, query_filter)

        if ok:
            passed += 1

    return passed / len(topk)


# =========================================================
# 1. query_transformer 테스트
# =========================================================

def test_query_transformer():
    print_header("1. query_transformer 테스트")

    for q in BOARDGAME_QUERIES[:1] + MURDER_QUERIES[:1]:
        category = "boardgame" if q in BOARDGAME_QUERIES else "murdermystery"
        result = query_transform(q["user_text"], q["group"], category)

        print_subheader(f"{q['name']} ({category})")
        print(f"  query_text:    {result['query_text']}")
        print(f"  query_filter:  {result['query_filter']}")
        print(f"  emotion_tags:  {result['emotion_tags']}")
        print(f"  anchor_titles: {result['anchor_titles']}")


# =========================================================
# 2. retriever 비교 테스트
# =========================================================

def test_retriever_comparison(queries, category):
    print_header(f"2. retriever 비교 — {category}")

    results_table = []

    for q in queries:
        transformed = query_transform(q["user_text"], q["group"], category)
        query_filter = transformed["query_filter"]
        query_vector = get_embedding(transformed["anchor_titles"], category)

        print_subheader(q["name"])

        # RRF
        t0 = time.time()
        rrf_items = retrieve(
            transformed["query_text"],
            query_filter,
            query_vector,
            category,
            topk=50,
        )
        rrf_time = time.time() - t0
        rrf_prec = precision_at_k(rrf_items, q["ground_truth"], k=10)
        rrf_cond = condition_pass_at_k(rrf_items, query_filter, category, k=10)

        # BM25
        t0 = time.time()
        bm25_items = retrieve_bm25(
            transformed["query_text"],
            query_filter,
            category,
            topk=50,
        )
        bm25_time = time.time() - t0
        bm25_prec = precision_at_k(bm25_items, q["ground_truth"], k=10)
        bm25_cond = condition_pass_at_k(bm25_items, query_filter, category, k=10)

        # Dense
        t0 = time.time()
        dense_items = retrieve_dense(
            query_vector,
            query_filter,
            category,
            topk=50,
        )
        dense_time = time.time() - t0
        dense_prec = precision_at_k(dense_items, q["ground_truth"], k=10)
        dense_cond = condition_pass_at_k(dense_items, query_filter, category, k=10)

        # Vanilla
        t0 = time.time()
        vanilla_items = retrieve_vanilla(
            query_filter,
            category,
            topk=50,
        )
        vanilla_time = time.time() - t0
        vanilla_prec = precision_at_k(vanilla_items, q["ground_truth"], k=10)
        vanilla_cond = condition_pass_at_k(vanilla_items, query_filter, category, k=10)

        print("\n  Title Ground Truth Precision@10")
        print(f"  {'방식':<12} {'P@10':>8} {'건수':>6} {'시간':>8}")
        print(f"  {'-' * 38}")

        for name, prec, items, elapsed in [
            ("RRF", rrf_prec, rrf_items, rrf_time),
            ("BM25", bm25_prec, bm25_items, bm25_time),
            ("Dense", dense_prec, dense_items, dense_time),
            ("Vanilla", vanilla_prec, vanilla_items, vanilla_time),
        ]:
            prec_str = f"{prec:.3f}" if prec is not None else "N/A"
            print(f"  {name:<12} {prec_str:>8} {len(items):>6} {elapsed:>7.2f}s")

        print("\n  Condition@10 — hard condition 만족 비율")
        print(f"  {'방식':<12} {'Cond@10':>8}")
        print(f"  {'-' * 24}")

        for name, cond in [
            ("RRF", rrf_cond),
            ("BM25", bm25_cond),
            ("Dense", dense_cond),
            ("Vanilla", vanilla_cond),
        ]:
            print(f"  {name:<12} {cond:>8.3f}")

        print("\n  RRF 상위 5개:")
        print_items(rrf_items, 5)

        results_table.append({
            "category": category,
            "query": q["name"],

            "rrf_prec": rrf_prec,
            "bm25_prec": bm25_prec,
            "dense_prec": dense_prec,
            "vanilla_prec": vanilla_prec,

            "rrf_cond": rrf_cond,
            "bm25_cond": bm25_cond,
            "dense_cond": dense_cond,
            "vanilla_cond": vanilla_cond,
        })

    return results_table


# =========================================================
# 3. tag_filter 테스트
# =========================================================

def test_tag_filter():
    print_header("3. tag_filter 테스트")

    # 보드게임: 2인 가벼운 파티게임
    q = BOARDGAME_QUERIES[1]
    transformed = query_transform(q["user_text"], q["group"], "boardgame")
    query_vector = get_embedding(transformed["anchor_titles"], "boardgame")

    items = retrieve(
        transformed["query_text"],
        transformed["query_filter"],
        query_vector,
        "boardgame",
        topk=20,
    )

    print_subheader(f"보드게임 필터 전: {len(items)}개")
    print_items(items, 3)

    filtered = filter_and_score(
        items,
        emotion_tags=transformed["emotion_tags"],
        horror_tolerance=q["group"].get("horror_tolerance", 2),
    )

    print_subheader(f"보드게임 필터 후: {len(filtered)}개")
    print_items(filtered, 3)

    # 머더미스터리: 6인 쉬운 입문
    q2 = MURDER_QUERIES[0]
    transformed2 = query_transform(q2["user_text"], q2["group"], "murdermystery")
    query_vector2 = get_embedding(transformed2["anchor_titles"], "murdermystery")

    items2 = retrieve(
        transformed2["query_text"],
        transformed2["query_filter"],
        query_vector2,
        "murdermystery",
        topk=20,
    )

    print_subheader(f"머더미스터리 필터 전: {len(items2)}개")
    print_items(items2, 3)

    filtered2 = filter_and_score(
        items2,
        emotion_tags=transformed2["emotion_tags"],
        horror_tolerance=q2["group"].get("horror_tolerance", 2),
    )

    print_subheader(f"머더미스터리 필터 후: {len(filtered2)}개")
    print_items(filtered2, 3)


# =========================================================
# 4. generator 테스트
# =========================================================

def _print_recommendation_result(result):
    recommendations = result.get("recommendations") or result.get("games") or []

    if not recommendations:
        print("  ⚠️ 추천 결과 없음")
        print(f"  answer: {result.get('answer')}")
        print(f"  next_question: {result.get('next_question') or result.get('follow_up_question')}")
        return

    for i, rec in enumerate(recommendations, 1):
        title = rec.get("title") or rec.get("name") or "?"
        reason = rec.get("reason") or rec.get("description") or ""
        score = rec.get("final_score") or rec.get("total_score") or rec.get("score")
        source = rec.get("source")

        print(f"  {i}. {title}")

        if source:
            print(f"     source: {source}")

        if score is not None:
            print(f"     score: {score}")

        if reason:
            print(f"     {reason}")

    follow_up = result.get("follow_up_question") or result.get("next_question")
    print(f"\n  ❓ 역질문: {follow_up}")


def test_generator():
    print_header("4. generator 테스트 (룰 기반)")

    # 보드게임
    q = BOARDGAME_QUERIES[0]
    transformed = query_transform(q["user_text"], q["group"], "boardgame")
    query_vector = get_embedding(transformed["anchor_titles"], "boardgame")

    items = retrieve(
        transformed["query_text"],
        transformed["query_filter"],
        query_vector,
        "boardgame",
        topk=10,
    )

    filtered = filter_and_score(
        items,
        transformed["emotion_tags"],
        horror_tolerance=q["group"].get("horror_tolerance", 2),
    )

    result = generate_without_api(
        filtered,
        q["group"],
        "boardgame",
        transformed["emotion_tags"],
    )

    print_subheader("보드게임 추천")
    _print_recommendation_result(result)

    # 머더미스터리
    q2 = MURDER_QUERIES[0]
    transformed2 = query_transform(q2["user_text"], q2["group"], "murdermystery")
    query_vector2 = get_embedding(transformed2["anchor_titles"], "murdermystery")

    items2 = retrieve(
        transformed2["query_text"],
        transformed2["query_filter"],
        query_vector2,
        "murdermystery",
        topk=10,
    )

    filtered2 = filter_and_score(
        items2,
        transformed2["emotion_tags"],
        horror_tolerance=q2["group"].get("horror_tolerance", 2),
    )

    result2 = generate_without_api(
        filtered2,
        q2["group"],
        "murdermystery",
        transformed2["emotion_tags"],
    )

    print_subheader("머더미스터리 추천")
    _print_recommendation_result(result2)


# =========================================================
# 5. graph E2E 테스트
# =========================================================

def test_graph_e2e():
    print_header("5. graph 파이프라인 E2E 테스트")

    test_cases = [
        ("boardgame", BOARDGAME_QUERIES[0]),
        ("murdermystery", MURDER_QUERIES[0]),
    ]

    for category, q in test_cases:
        user_text = q["user_text"]
        group = q["group"]

        print_subheader(f'{category}: "{user_text}"')

        t0 = time.time()

        try:
            result = run_pipeline(
                user_text=user_text,
                group=group,
                category=category,
                use_api=False,
            )
        except TypeError:
            result = run_pipeline(
                user_text,
                group,
                category,
                use_api=False,
            )

        elapsed = time.time() - t0

        games = result.get("games") or result.get("recommendations") or []
        next_question = (
            result.get("next_question")
            or result.get("follow_up_question")
            or ""
        )

        print(f"  추천 {len(games)}개 생성 ({elapsed:.2f}s)")

        if not games:
            print("  ⚠️ graph E2E 결과가 비어 있음")
            print("  result keys:", list(result.keys()))
            print("  raw result:", result)
        else:
            for i, rec in enumerate(games[:5], 1):
                title = rec.get("title") or rec.get("name") or "?"
                reason = rec.get("reason") or rec.get("description") or ""
                score = rec.get("final_score") or rec.get("total_score") or rec.get("score")
                source = rec.get("source")

                print(f"  {i}. {title}")

                if source:
                    print(f"     source: {source}")

                if score is not None:
                    print(f"     score: {score}")

                if reason:
                    print(f"     reason: {reason[:80]}...")

        print(f"  ❓ {next_question}")


# =========================================================
# 6. 종합 평가표
# =========================================================

def _avg(rows, key):
    vals = [
        r[key]
        for r in rows
        if r.get(key) is not None
    ]

    return sum(vals) / len(vals) if vals else 0.0


def print_summary(bg_results, mm_results):
    print_header("6. 종합 평가 점수 비교표")

    all_results = bg_results + mm_results

    print("\n  6-1. Title Ground Truth Precision@10")
    print(f"\n  {'쿼리':<30} {'RRF':>8} {'BM25':>8} {'Dense':>8} {'Vanilla':>8}")
    print(f"  {'-' * 66}")

    for r in all_results:
        rrf = f"{r['rrf_prec']:.3f}" if r["rrf_prec"] is not None else "N/A"
        bm25 = f"{r['bm25_prec']:.3f}" if r["bm25_prec"] is not None else "N/A"
        dense = f"{r['dense_prec']:.3f}" if r["dense_prec"] is not None else "N/A"
        vanilla = f"{r['vanilla_prec']:.3f}" if r["vanilla_prec"] is not None else "N/A"

        print(
            f"  {r['query']:<30} "
            f"{rrf:>8} "
            f"{bm25:>8} "
            f"{dense:>8} "
            f"{vanilla:>8}"
        )

    print(f"  {'-' * 66}")
    print(
        f"  {'전체 평균':<30} "
        f"{_avg(all_results, 'rrf_prec'):>8.3f} "
        f"{_avg(all_results, 'bm25_prec'):>8.3f} "
        f"{_avg(all_results, 'dense_prec'):>8.3f} "
        f"{_avg(all_results, 'vanilla_prec'):>8.3f}"
    )

    print("\n  6-2. Condition@10 — hard condition 만족 점수")
    print(f"\n  {'도메인':<18} {'RRF':>8} {'BM25':>8} {'Dense':>8} {'Vanilla':>8}")
    print(f"  {'-' * 58}")

    for domain, rows in [
        ("boardgame", bg_results),
        ("murdermystery", mm_results),
        ("overall", all_results),
    ]:
        print(
            f"  {domain:<18} "
            f"{_avg(rows, 'rrf_cond'):>8.3f} "
            f"{_avg(rows, 'bm25_cond'):>8.3f} "
            f"{_avg(rows, 'dense_cond'):>8.3f} "
            f"{_avg(rows, 'vanilla_cond'):>8.3f}"
        )

    print("\n  참고:")
    print("  - Precision@10은 ground_truth 제목이 top10에 포함되었는지 보는 지표입니다.")
    print("  - Condition@10은 top10 결과가 hard condition을 만족하는지 보는 지표입니다.")
    print("  - 보드게임 Condition@10은 인원/시간 조건을 기준으로 계산합니다.")
    print("  - 머더미스터리 Condition@10은 인원/시간/scene_category 조건을 기준으로 계산합니다.")
    print("  - ground_truth가 비어 있는 쿼리는 Precision@10에서 N/A로 표시되지만, Condition@10은 계산됩니다.")


# =========================================================
# 메인
# =========================================================

if __name__ == "__main__":
    print("\n🚀 Nolit RAG 파이프라인 통합 평가 시작\n")
    total_start = time.time()

    # 1. query_transformer
    test_query_transformer()

    # 2. retriever 비교
    bg_results = test_retriever_comparison(BOARDGAME_QUERIES, "boardgame")
    mm_results = test_retriever_comparison(MURDER_QUERIES, "murdermystery")

    # 3. tag_filter
    test_tag_filter()

    # 4. generator
    test_generator()

    # 5. graph E2E
    test_graph_e2e()

    # 6. 종합 비교
    print_summary(bg_results, mm_results)

    total_elapsed = time.time() - total_start
    print(f"\n✅ 전체 평가 완료 ({total_elapsed:.1f}s)")