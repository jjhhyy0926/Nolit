"""
yoonha_eval_pipeline.py
전체 RAG 파이프라인 통합 평가

테스트 항목:
    1. query_transformer 변환 확인
    2. retriever 검색 결과 확인 (RRF / BM25 / Dense / Vanilla)
    3. tag_filter 필터링 확인
    4. generator 생성 확인 (룰 기반)
    5. graph 파이프라인 E2E 확인
    6. Precision@K 비교 (RRF vs BM25 vs Dense vs Vanilla)
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
        "ground_truth": ["Brass: Birmingham", "브라스: 버밍엄", "Twilight Struggle"],
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
        "ground_truth": ["Codenames", "Dixit", "코드네임"],
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
        "ground_truth": ["Pandemic", "Spirit Island", "팬데믹"],
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
        "ground_truth": ["구두룡 저택의 살인", "몇 번이고 푸른 달에 불을 붙였다"],
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
        "ground_truth": ["구두룡 저택의 살인"],
    },
]


# =========================================================
# 공통 유틸
# =========================================================

def precision_at_k(items, ground_truth, k=10):
    if not ground_truth:
        return None

    pred_titles = [
        item.get("title", item.get("name", ""))
        for item in items[:k]
    ]

    hits = sum(1 for gt in ground_truth if gt in pred_titles)
    return hits / k


def print_header(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_subheader(text):
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


def _num(value, default=None):
    if value is None:
        return default

    if isinstance(value, (int, float)):
        try:
            if value != value:
                return default
        except Exception:
            pass
        return value

    try:
        import re
        m = re.search(r"\d+(?:\.\d+)?", str(value))
        return float(m.group(0)) if m else default
    except Exception:
        return default


def _players_pass(item, players):
    if players is None:
        return True

    min_p = _num(item.get("min_players"), 0)
    max_p = _num(item.get("max_players"), 999)

    return min_p <= players <= max_p


def _time_pass(item, max_time, category):
    if max_time is None:
        return True

    if category == "boardgame":
        if item.get("source") == "boardlife":
            item_time = _num(item.get("max_time"), None)
        else:
            item_time = _num(item.get("playing_time"), None)
    else:
        item_time = _num(item.get("max_time"), None)

        if item_time is None:
            item_time = _num(item.get("play_time"), None)

    if item_time is None or item_time <= 0:
        return True

    return item_time <= max_time


def validate_hard_filter_results(items, query_filter, category, label, k=10):
    print_subheader(f"하드필터 검증 — {label}")

    players = query_filter.get("players")
    max_time = query_filter.get("playing_time") or query_filter.get("max_time")

    failures = []

    for idx, item in enumerate(items[:k], 1):
        title = item.get("title", item.get("name", "?"))

        if not _players_pass(item, players):
            failures.append(
                f"{idx}. {title} — 인원 조건 실패 "
                f"(query={players}, item={item.get('min_players')}~{item.get('max_players')})"
            )

        if not _time_pass(item, max_time, category):
            item_time = (
                item.get("max_time")
                or item.get("playing_time")
                or item.get("play_time")
            )
            failures.append(
                f"{idx}. {title} — 시간 조건 실패 "
                f"(query<={max_time}, item={item_time})"
            )

    if failures:
        print("  ⚠️ FAIL")
        for f in failures:
            print(f"  - {f}")
        return False

    print(f"  ✅ PASS: 상위 {min(k, len(items))}개 모두 하드필터 통과")
    return True


def validate_preference_direction(items, query_filter, category, label, k=10):
    print_subheader(f"선호 방향성 검증 — {label}")

    topk = items[:k]

    if not topk:
        print("  ⚠️ 결과 없음")
        return False

    if category == "boardgame":
        pref = query_filter.get("weight_pref")

        weights = [
            _num(item.get("weight"), None)
            for item in topk
            if _num(item.get("weight"), None) is not None
        ]

        if not pref or not weights:
            print("  - weight_pref 또는 weight 데이터 없음")
            return True

        avg_weight = sum(weights) / len(weights)

        print(f"  weight_pref: {pref}")
        print(f"  top{k} 평균 weight: {avg_weight:.2f}")

        if pref == "light" and avg_weight > 3.2:
            print("  ⚠️ 확인 필요: 가벼운 게임 쿼리인데 평균 weight가 높음")
            return False

        if pref == "heavy" and avg_weight < 2.7:
            print("  ⚠️ 확인 필요: 무거운 게임 쿼리인데 평균 weight가 낮음")
            return False

        print("  ✅ PASS: weight 방향성 정상 범위")
        return True

    if category == "murdermystery":
        difficulty_pref = query_filter.get("difficulty_pref")
        horror_pref = query_filter.get("horror_pref")

        difficulties = [
            _num(item.get("difficulty"), None)
            for item in topk
            if _num(item.get("difficulty"), None) is not None
        ]

        horrors = [
            _num(item.get("horror"), None)
            for item in topk
            if _num(item.get("horror"), None) is not None
        ]

        if difficulty_pref and difficulties:
            avg_diff = sum(difficulties) / len(difficulties)
            print(f"  difficulty_pref: {difficulty_pref}")
            print(f"  top{k} 평균 difficulty: {avg_diff:.2f}")

        if horror_pref and horrors:
            avg_horror = sum(horrors) / len(horrors)
            print(f"  horror_pref: {horror_pref}")
            print(f"  top{k} 평균 horror: {avg_horror:.2f}")

            if horror_pref == "low" and avg_horror > 2.5:
                print("  ⚠️ 확인 필요: 안 무서운 쿼리인데 horror 평균이 높음")
                return False

        print("  ✅ PASS: difficulty/horror 방향성 점검 완료")
        return True

    return True


def validate_source_weight(items, query_filter, label, k=10):
    print_subheader(f"소스 가중치 검증 — {label}")

    source_pref = query_filter.get("source_pref") or query_filter.get("source_preference")

    if source_pref not in {"korean", "boardlife", None}:
        print(f"  - source_pref={source_pref}, boardlife 우선 검증 생략")
        return True

    topk = items[:k]

    if not topk:
        print("  ⚠️ 결과 없음")
        return False

    boardlife_count = sum(1 for item in topk if item.get("source") == "boardlife")
    ratio = boardlife_count / len(topk)

    print(f"  top{k} boardlife 비율: {boardlife_count}/{len(topk)} = {ratio:.2f}")

    if ratio == 0:
        print("  ⚠️ 확인 필요: 한국어 쿼리인데 boardlife 결과가 상위권에 없음")
        return False

    print("  ✅ PASS: boardlife 결과 반영됨")
    return True


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
        query_vector = get_embedding(transformed["anchor_titles"], category)

        print_subheader(q["name"])

        # RRF
        t0 = time.time()
        rrf_items = retrieve(
            transformed["query_text"],
            transformed["query_filter"],
            query_vector,
            category,
            topk=50,
        )
        rrf_time = time.time() - t0
        rrf_prec = precision_at_k(rrf_items, q["ground_truth"], k=10)

        # BM25
        t0 = time.time()
        bm25_items = retrieve_bm25(
            transformed["query_text"],
            transformed["query_filter"],
            category,
            topk=50,
        )
        bm25_time = time.time() - t0
        bm25_prec = precision_at_k(bm25_items, q["ground_truth"], k=10)

        # Dense
        t0 = time.time()
        dense_items = retrieve_dense(
            query_vector,
            transformed["query_filter"],
            category,
            topk=50,
        )
        dense_time = time.time() - t0
        dense_prec = precision_at_k(dense_items, q["ground_truth"], k=10)

        # Vanilla
        t0 = time.time()
        vanilla_items = retrieve_vanilla(
            transformed["query_filter"],
            category,
            topk=50,
        )
        vanilla_time = time.time() - t0
        vanilla_prec = precision_at_k(vanilla_items, q["ground_truth"], k=10)

        print(f"\n  {'방식':<12} {'P@10':>8} {'건수':>6} {'시간':>8}")
        print(f"  {'-' * 38}")

        for name, prec, items, elapsed in [
            ("RRF", rrf_prec, rrf_items, rrf_time),
            ("BM25", bm25_prec, bm25_items, bm25_time),
            ("Dense", dense_prec, dense_items, dense_time),
            ("Vanilla", vanilla_prec, vanilla_items, vanilla_time),
        ]:
            prec_str = f"{prec:.3f}" if prec is not None else "N/A"
            print(f"  {name:<12} {prec_str:>8} {len(items):>6} {elapsed:>7.2f}s")

        print("\n  RRF 상위 5개:")
        print_items(rrf_items, 5)

        validate_hard_filter_results(
            rrf_items,
            transformed["query_filter"],
            category,
            q["name"],
            k=10,
        )

        validate_preference_direction(
            rrf_items,
            transformed["query_filter"],
            category,
            q["name"],
            k=10,
        )

        if category == "boardgame":
            validate_source_weight(
                rrf_items,
                transformed["query_filter"],
                q["name"],
                k=10,
            )

        results_table.append({
            "query": q["name"],
            "rrf_prec": rrf_prec,
            "bm25_prec": bm25_prec,
            "dense_prec": dense_prec,
            "vanilla_prec": vanilla_prec,
        })

    return results_table


# =========================================================
# 3. tag_filter 테스트
# =========================================================

def test_tag_filter():
    print_header("3. tag_filter 테스트")

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

    print_subheader(f"필터 전: {len(items)}개")
    print_items(items, 3)

    filtered = filter_and_score(
        items,
        emotion_tags=transformed["emotion_tags"],
        horror_tolerance=q["group"].get("horror_tolerance", 2),
    )

    print_subheader(f"필터 후: {len(filtered)}개")
    print_items(filtered, 3)

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

    filtered = filter_and_score(items, transformed["emotion_tags"])

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
        next_question = result.get("next_question") or result.get("follow_up_question") or ""

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
# 6. 종합 Precision 비교표
# =========================================================

def print_summary(bg_results, mm_results):
    print_header("6. 종합 Precision@10 비교표")

    print(f"\n  {'쿼리':<30} {'RRF':>8} {'BM25':>8} {'Dense':>8} {'Vanilla':>8}")
    print(f"  {'-' * 66}")

    all_results = bg_results + mm_results

    for r in all_results:
        rrf = f"{r['rrf_prec']:.3f}" if r["rrf_prec"] is not None else "N/A"
        bm25 = f"{r['bm25_prec']:.3f}" if r["bm25_prec"] is not None else "N/A"
        dense = f"{r['dense_prec']:.3f}" if r["dense_prec"] is not None else "N/A"
        vanilla = f"{r['vanilla_prec']:.3f}" if r["vanilla_prec"] is not None else "N/A"

        print(f"  {r['query']:<30} {rrf:>8} {bm25:>8} {dense:>8} {vanilla:>8}")

    def avg(key):
        vals = [r[key] for r in all_results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    print(f"  {'-' * 66}")
    print(
        f"  {'평균':<30} "
        f"{avg('rrf_prec'):>8.3f} "
        f"{avg('bm25_prec'):>8.3f} "
        f"{avg('dense_prec'):>8.3f} "
        f"{avg('vanilla_prec'):>8.3f}"
    )


# =========================================================
# 메인
# =========================================================

if __name__ == "__main__":
    print("\n🚀 Nolit RAG 파이프라인 통합 평가 시작\n")
    total_start = time.time()

    test_query_transformer()

    bg_results = test_retriever_comparison(BOARDGAME_QUERIES, "boardgame")
    mm_results = test_retriever_comparison(MURDER_QUERIES, "murdermystery")

    test_tag_filter()
    test_generator()
    test_graph_e2e()

    print_summary(bg_results, mm_results)

    total_elapsed = time.time() - total_start
    print(f"\n✅ 전체 평가 완료 ({total_elapsed:.1f}s)")