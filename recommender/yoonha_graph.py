"""
recommender/yoonha_graph.py

LangGraph 기반 RAG 추천 파이프라인.

====================================================================
[역할]
    사용자의 자연어 요청을 받아 보드게임 또는 머더미스터리를 추천한다.
    입력 정제 → 조건 충분성 판단 → 쿼리 변환 → 하이브리드 검색 →
    태그 필터링 → 텍스트 생성의 6단계 파이프라인을 LangGraph로 구성한다.

[외부 호출 예시]
    from recommender.graph import graph

    result = graph.invoke({
        "query": "4명이서 할 보드게임",
        "category": "boardgame"
    })

[출력 스펙]
    {
        "answer":        "추천 텍스트",
        "games":         [...],
        "next_question": "..."
    }

[내부 흐름]
    normalize_input
    → check_sufficiency
        ├─ 부족함 → clarify → END
        └─ 충분함 → query_transform → retrieve → tag_filter → generate → END

[설계 문서와의 대응]
    - 기획서 §4: 파이프라인 전체 흐름 (normalize → check → retrieve → generate)
    - 기획서 §5: BM25 + FAISS 하이브리드 검색 (node_retrieve 참조)
    - 필터 문서 §공통: headcount 필수 조건 판단 (node_check_sufficiency 참조)
    - 필터 문서 §가중치: emotion_tags 매칭 (node_tag_filter 참조)

[개선 포인트]
    1. check_sufficiency의 필수 조건이 query/category/headcount로 고정됨
       → 카테고리별로 필수 조건을 다르게 설정할 여지 있음
    2. use_api=False 시 로컬 fallback만 사용 → OpenAI 없이 품질 저하
    3. _build_next_question이 단일 missing_field만 처리
       → 복수 조건 누락 시 한 번에 물어보는 방식으로 개선 가능
    4. node_retrieve에서 예외를 전부 catch → 오류 원인 진단이 어려울 수 있음
====================================================================
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------
# 직접 실행 / 패키지 import 모두 지원하기 위한 path 보정
# ---------------------------------------------------------------------
# 스크립트로 직접 실행(python yoonha_graph.py)하거나
# 패키지로 import 할 때 모두 PROJECT_ROOT를 sys.path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Type 정의
# =====================================================================

Category = Literal["boardgame", "murdermystery"]


class GraphInput(TypedDict, total=False):
    """
    외부에서 graph.invoke()로 넣는 입력 스펙.

    필수:
        query:    사용자 자연어 요청 ("4명이서 할 보드게임" 등)
        category: "boardgame" | "murdermystery"

    선택:
        group:   이미 파싱된 그룹 조건 (headcount, play_time 등을 dict로 직접 전달)
        use_api: OpenAI API 사용 여부 (False이면 룰 기반 fallback 사용)
    """

    query: str
    user_text: str
    category: str
    group: dict[str, Any]
    use_api: bool


class GraphOutput(TypedDict):
    """
    외부로 반환되는 최종 출력 스펙.

    answer:        추천 텍스트 (자연어)
    games:         추천 게임 리스트 (각 항목은 title, reason, matched_tags 등 포함)
    next_question: 조건 보완을 위한 역질문 (조건이 충분하면 빈 문자열)
    """

    answer: str
    games: list[dict[str, Any]]
    next_question: str


class PipelineState(TypedDict, total=False):
    """
    LangGraph 내부 state.

    [흐름별 필드 설명]
        - 외부 입력:      query, user_text, category, group, use_api
        - 조건 판단:      is_sufficient, missing_fields
        - RAG 중간 결과:  query_text, query_filter, emotion_tags,
                          anchor_titles, retrieved_items, filtered_items
        - 에러/진단:      retrieve_error, generate_error
        - 최종 출력:      result, answer, games, next_question

    total=False이므로 모든 필드가 선택적 → 초기화 시 없는 키도 허용
    """

    # ── 외부 입력 ──
    query: str
    user_text: str
    category: str
    group: dict[str, Any]
    use_api: bool

    # ── 조건 판단 ──
    is_sufficient: bool         # True: RAG 진행, False: clarify 노드로 분기
    missing_fields: list[str]   # 부족한 필드명 목록 (e.g. ["headcount", "category"])

    # ── RAG 중간 결과 ──
    query_text: str                     # BM25 검색용 자연어 텍스트
    query_filter: dict[str, Any]        # 하드 필터 조건 (players, max_time 등)
    emotion_tags: list[str]             # 감정 태그 목록 (e.g. ["신남", "긴장감"])
    anchor_titles: list[str]            # FAISS 앵커 타이틀 (유사 작품명)
    retrieved_items: list[dict[str, Any]]   # 검색 결과 원본
    filtered_items: list[dict[str, Any]]    # 태그 필터 후 정제 결과

    # ── 에러/진단 ──
    retrieve_error: str     # 검색 단계 예외 메시지
    generate_error: str     # 생성 단계 예외 메시지

    # ── 최종 출력 ──
    result: dict[str, Any]          # 노드 간 전달용 중간 dict
    answer: str
    games: list[dict[str, Any]]
    next_question: str


# =====================================================================
# 입력 파싱 유틸
# =====================================================================

# 지원하는 카테고리 집합 — check_sufficiency에서 유효성 검사에 사용
SUPPORTED_CATEGORIES = {"boardgame", "murdermystery"}

# 카테고리 별칭 → 내부 표준값 매핑
# 한국어/영어/약어/오타 등 다양한 입력을 허용하기 위해 alias를 폭넓게 정의
CATEGORY_ALIASES = {
    "boardgame": "boardgame",
    "boardgames": "boardgame",
    "board_game": "boardgame",
    "board-game": "boardgame",
    "board": "boardgame",
    "보드게임": "boardgame",
    "보드": "boardgame",
    "murdermystery": "murdermystery",
    "murder_mystery": "murdermystery",
    "murder-mystery": "murdermystery",
    "murder": "murdermystery",
    "mm": "murdermystery",
    "crime": "murdermystery",
    "crimescene": "murdermystery",
    "crime_scene": "murdermystery",
    "머더미스터리": "murdermystery",
    "머더": "murdermystery",
    "크라임씬": "murdermystery",
    "크라임": "murdermystery",
}

# 한국어 수사 → 숫자 변환 테이블
# _extract_headcount에서 "두 명", "넷이서" 등의 표현 처리에 사용
KOREAN_NUMBER_MAP = {
    "한": 1,
    "하나": 1,
    "둘": 2,
    "두": 2,
    "셋": 3,
    "세": 3,
    "넷": 4,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
}

# BGG 카테고리 키워드 → 내부 표준값 매핑
# "전략 보드게임"처럼 query에서 카테고리를 명시한 경우 추출
BOARDGAME_CATEGORY_KEYWORDS = {
    "전략": "Strategy",
    "경제": "Economic",
    "파티": "Party",
    "전쟁": "War",
    "가족": "Family",
    "패밀리": "Family",
    "추상": "Abstract",
    "협력": "Cooperative",
    "협동": "Cooperative",
    "추리": "Deduction",
    "카드": "Card Game",
    "테마": "Thematic",
}

# BGG 메커니즘 키워드 → 내부 표준값 매핑
# "일꾼 배치", "덱빌딩" 등 메커니즘 선호를 query에서 직접 명시한 경우 추출
BOARDGAME_MECHANISM_KEYWORDS = {
    "일꾼": "Worker Placement",
    "워커": "Worker Placement",
    "덱빌딩": "Deck Building",
    "덱 빌딩": "Deck Building",
    "엔진": "Engine Building",
    "엔진빌딩": "Engine Building",
    "지역장악": "Area Control",
    "영역": "Area Control",
    "마켓": "Market",
    "시장": "Market",
    "드래프팅": "Drafting",
    "드래프트": "Drafting",
    "협력형": "Cooperative Game",
}


def _normalize_category(raw_category: str | None, query: str) -> str:
    """
    외부 category 값을 내부 표준 category로 정규화한다.
    category가 없거나 alias에 없는 값이면 query에서 키워드로 추론한다.

    [우선순위]
        1) CATEGORY_ALIASES에 정확히 매칭되는 값
        2) query에 "머더/크라임/murder/crime" 포함 → murdermystery
        3) query에 "보드/board" 포함 → boardgame
        4) 모두 해당 없으면 raw_category 그대로 반환 (check_sufficiency에서 탈락)
    """

    value = (raw_category or "").strip().lower()

    # 1) 정확히 알려진 alias 매칭
    if value in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[value]

    # 2~3) query에서 카테고리 키워드 추론
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in ["머더", "크라임", "murder", "crime"]):
        return "murdermystery"

    if any(keyword in query_lower for keyword in ["보드", "board"]):
        return "boardgame"

    return value


def _extract_headcount(query: str) -> int | None:
    """
    query에서 인원 수를 추출한다.

    [지원 패턴]
        숫자형: "4명이서", "4인 보드게임"
        한국어 수사형: "네 명이서", "넷이서", "두 명"

    Returns:
        int: 파싱된 인원 수
        None: 인원 정보를 찾을 수 없는 경우
    """

    if not query:
        return None

    # 숫자 + 명/인 패턴 (예: "4명", "10인")
    digit_match = re.search(r"(\d{1,2})\s*(명|인)", query)
    if digit_match:
        return int(digit_match.group(1))

    # 한국어 수사 패턴 (예: "네 명", "넷이서", "두명이서")
    for word, number in KOREAN_NUMBER_MAP.items():
        patterns = [
            rf"{word}\s*명",
            rf"{word}\s*인",
            rf"{word}이서",
            rf"{word}명이서",
        ]
        if any(re.search(pattern, query) for pattern in patterns):
            return number

    return None


def _extract_play_time(query: str) -> int | None:
    """
    query에서 최대 플레이 시간을 분 단위로 추출한다.

    [지원 패턴]
        "2시간 안에" → 120
        "90분 이하"  → 90
        "1시간 반"   → 90  (시간 + 반 조합)
        "1시간 30분" → 90  (시간 + 분 조합)

    Returns:
        int: 분 단위 플레이 시간
        None: 시간 정보를 찾을 수 없는 경우
    """

    if not query:
        return None

    hour_match = re.search(r"(\d{1,2})\s*시간", query)
    minute_match = re.search(r"(\d{1,3})\s*분", query)

    total_minutes = 0

    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60

    # "1시간 반" → hour_match 있을 때만 +30분 적용
    if "반" in query and hour_match:
        total_minutes += 30

    if minute_match:
        total_minutes += int(minute_match.group(1))

    return total_minutes or None


def _extract_weight_pref(query: str) -> str | None:
    """
    보드게임 난이도 선호를 query에서 추출한다.

    [매핑]
        "쉬운", "가벼운", "초보" 등 → "light"
        "보통", "중급", "적당"  등 → "medium"
        "어려운", "헤비", "복잡" 등 → "heavy"

    Returns:
        "light" | "medium" | "heavy" | None
    """

    if not query:
        return None

    light_keywords = [
        "쉬운",
        "쉽게",
        "가벼운",
        "가볍게",
        "간단",
        "입문",
        "초보",
        "룰 쉬운",
        "룰이 쉬운",
        "부담없는",
        "부담 없는",
    ]

    medium_keywords = [
        "보통",
        "중간",
        "중급",
        "적당",
        "무난",
    ]

    heavy_keywords = [
        "어려운",
        "어렵",
        "고난도",
        "고난이도",
        "헤비",
        "무거운",
        "복잡",
        "빡센",
        "깊이있는",
        "깊이 있는",
    ]

    # 순서 주의: light → heavy → medium 순으로 검사
    # "가볍지만 전략적인"처럼 light + heavy가 동시에 있을 경우 light 우선
    if any(keyword in query for keyword in light_keywords):
        return "light"

    if any(keyword in query for keyword in heavy_keywords):
        return "heavy"

    if any(keyword in query for keyword in medium_keywords):
        return "medium"

    return None


def _extract_horror_tolerance(query: str) -> int | None:
    """
    머더미스터리 공포 수용도를 query에서 추출한다.

    [매핑]
        "공포 싫어", "무서운 거 싫어" 등 → 0 (공포 불가)
        "약간 무서운", "살짝 공포"   등 → 1 (약간 가능)
        "공포 괜찮아", "호러 가능"   등 → 2 (가능)

    Returns:
        0: 공포 불가
        1: 약간 가능
        2: 가능
        None: 명시적 의사 표현 없음
    """

    if not query:
        return None

    horror_no_keywords = [
        "공포 싫",
        "공포는 싫",
        "공포 못",
        "공포 불가",
        "무서운 거 싫",
        "무서운건 싫",
        "무서운 것 싫",
        "겁 많",
        "겁많",
        "안 무서운",
        "안무서운",
        "공포 없는",
        "공포없",
    ]

    horror_low_keywords = [
        "약간 무서운",
        "살짝 무서운",
        "공포 조금",
        "조금 무서운",
    ]

    horror_ok_keywords = [
        "공포 괜찮",
        "무서운 거 괜찮",
        "호러 괜찮",
        "무서워도 괜찮",
    ]

    if any(keyword in query for keyword in horror_no_keywords):
        return 0

    if any(keyword in query for keyword in horror_low_keywords):
        return 1

    if any(keyword in query for keyword in horror_ok_keywords):
        return 2

    return None


def _extract_relation(query: str) -> str | None:
    """
    그룹 관계 유형을 query에서 추출한다.

    [매핑]
        "처음", "소개팅", "어색"    → "first_meeting"
        "데이트", "커플", "연인"    → "couple"
        "친구", "동창", "모임"     → "friend"
        "회식", "직장", "팀빌딩"   → "coworker"

    Returns:
        "first_meeting" | "couple" | "friend" | "coworker" | None
    """

    if not query:
        return None

    if any(keyword in query for keyword in ["처음", "첫만남", "첫 만남", "소개팅", "어색"]):
        return "first_meeting"

    if any(keyword in query for keyword in ["데이트", "커플", "연인", "남자친구", "여자친구"]):
        return "couple"

    if any(keyword in query for keyword in ["친구", "동창", "동기", "모임"]):
        return "friend"

    if any(keyword in query for keyword in ["회식", "직장", "회사", "동료", "팀빌딩", "워크샵"]):
        return "coworker"

    return None


def _extract_boardgame_category(query: str) -> str | None:
    """
    query에서 보드게임 BGG 카테고리를 추출한다.
    BOARDGAME_CATEGORY_KEYWORDS 테이블을 순서대로 탐색하며 첫 매칭을 반환한다.

    Returns:
        BGG 카테고리 문자열 (e.g. "Strategy") | None
    """

    if not query:
        return None

    for keyword, category in BOARDGAME_CATEGORY_KEYWORDS.items():
        if keyword in query:
            return category

    return None


def _extract_boardgame_mechanism(query: str) -> str | None:
    """
    query에서 보드게임 메커니즘을 추출한다.
    BOARDGAME_MECHANISM_KEYWORDS 테이블을 순서대로 탐색하며 첫 매칭을 반환한다.

    Returns:
        BGG 메커니즘 문자열 (e.g. "Worker Placement") | None
    """

    if not query:
        return None

    for keyword, mechanism in BOARDGAME_MECHANISM_KEYWORDS.items():
        if keyword in query:
            return mechanism

    return None


def _merge_group_from_query(
    query: str,
    category: str,
    group: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    외부 group dict와 query에서 파싱한 조건을 병합한다.

    [우선순위]
        명시적으로 전달된 group 값 > query 파싱 결과
        → group에 이미 존재하는 키는 덮어쓰지 않는다.

    [카테고리별 추출 항목]
        공통:    headcount, play_time, weight_pref, horror_tolerance, relation
        boardgame 전용: category(BGG), mechanism
    """

    merged: dict[str, Any] = dict(group or {})

    # ── 공통 조건 추출 ──
    if "headcount" not in merged:
        headcount = _extract_headcount(query)
        if headcount is not None:
            merged["headcount"] = headcount

    if "play_time" not in merged:
        play_time = _extract_play_time(query)
        if play_time is not None:
            merged["play_time"] = play_time

    if "weight_pref" not in merged:
        weight_pref = _extract_weight_pref(query)
        if weight_pref is not None:
            merged["weight_pref"] = weight_pref

    if "horror_tolerance" not in merged:
        horror_tolerance = _extract_horror_tolerance(query)
        if horror_tolerance is not None:
            merged["horror_tolerance"] = horror_tolerance

    if "relation" not in merged:
        relation = _extract_relation(query)
        if relation is not None:
            merged["relation"] = relation

    # ── boardgame 전용 조건 추출 ──
    if category == "boardgame":
        if "category" not in merged:
            boardgame_category = _extract_boardgame_category(query)
            if boardgame_category is not None:
                merged["category"] = boardgame_category

        if "mechanism" not in merged:
            mechanism = _extract_boardgame_mechanism(query)
            if mechanism is not None:
                merged["mechanism"] = mechanism

    return merged


def _build_next_question(
    category: str,
    group: dict[str, Any],
    missing_fields: list[str] | None = None,
) -> str:
    """
    현재 group 조건에서 가장 우선 보완이 필요한 항목에 대한 역질문을 생성한다.

    [판단 우선순위]
        1) query 자체가 없음
        2) category 미확인
        3) headcount 없음
        4) boardgame: weight_pref → play_time → relation 순
        5) murdermystery: horror_tolerance → play_time → relation 순
        6) 위 조건 모두 충족 시 → 추가 선호 질문

    [설계 의도]
        clarify 노드뿐 아니라 generate 노드에서도 호출하여
        추천 결과와 함께 다음 보완 질문을 함께 반환한다.
    """

    missing_fields = missing_fields or []

    # ── 필수 누락 필드 처리 ──
    if "query" in missing_fields:
        return "어떤 활동을 찾고 계신지 알려주세요. 예: 4명이서 할 보드게임"

    if "category" in missing_fields:
        return "보드게임과 머더미스터리 중 어떤 활동을 추천받고 싶으신가요?"

    if "headcount" in missing_fields or not group.get("headcount"):
        return "몇 명이서 함께할 예정인가요?"

    # ── boardgame 선택 조건 보완 ──
    if category == "boardgame":
        if not group.get("weight_pref"):
            return "게임 난이도는 어느 정도가 좋으세요? 가벼운 입문용, 보통, 어려운 전략 게임 중에서 골라주세요."

        if not group.get("play_time"):
            return "플레이 시간은 어느 정도가 적당한가요? 예: 30분, 1시간, 2시간"

        if not group.get("relation"):
            return "함께하는 분들과의 관계가 어떻게 되나요? 친구, 연인, 직장동료, 첫 만남 중에 알려주세요."

    # ── murdermystery 선택 조건 보완 ──
    if category == "murdermystery":
        if group.get("horror_tolerance") is None:
            return "공포 요소는 괜찮으신가요? 공포 불가, 약간 가능, 괜찮음 중에서 알려주세요."

        if not group.get("play_time"):
            return "플레이 시간은 어느 정도가 적당한가요? 예: 2시간, 3시간"

        if not group.get("relation"):
            return "함께하는 분들과의 관계가 어떻게 되나요? 친구, 연인, 직장동료, 첫 만남 중에 알려주세요."

    # ── 모든 핵심 조건 충족 시 추가 선호 질문 ──
    return "추가로 피하고 싶은 요소나 선호하는 분위기가 있나요?"


def _normalize_result(result: dict[str, Any] | None) -> dict[str, Any]:
    """
    generator 결과를 최종 출력 스펙(GraphOutput)으로 맞춘다.

    [목적]
        generate 노드가 어떤 경로(API / fallback)로 결과를 만들든
        항상 동일한 키 구조로 반환되도록 보장한다.
        None이거나 키가 없는 경우 빈 값으로 채운다.
    """

    result = result or {}

    return {
        "answer": result.get("answer", "") or "",
        "games": result.get("games", []) or [],
        "next_question": result.get("next_question", "") or "",
    }


# =====================================================================
# 룰 기반 generator fallback
# =====================================================================
def _generate_without_api_local(
    items: list[dict[str, Any]],
    group: dict[str, Any],
    category: str,
    emotion_tags: list[str] | None = None,
    max_items: int = 5,
    retrieve_error: str = "",
) -> dict[str, Any]:
    """
    OpenAI API 없이 동작하는 로컬 fallback generator.

    [사용 시점]
        1) use_api=False 로 명시적 비활성화
        2) yoonha_generator.generate() 호출 중 예외 발생

    [로직]
        retrieved_items를 순회하며 group 조건(인원, 난이도, 태그)과 비교하여
        각 게임별 reason 문자열을 조립한다.
        OpenAI를 사용하지 않으므로 자연스러운 문장 생성은 어렵지만,
        graph.invoke()가 항상 결과를 반환하도록 안전망 역할을 한다.

    [개선 포인트]
        - reason 조립이 단순 분기라 문장이 딱딱함
        - 인원 범위를 벗어나는 아이템도 retrieved_items에 포함될 수 있음
          → tag_filter에서 하드 필터링이 선행되어야 함
    """

    emotion_tags = emotion_tags or []
    games: list[dict[str, Any]] = []

    # ── 검색 결과가 없는 경우 ──
    if not items:
        if retrieve_error:
            # 데이터/인덱스가 연결되지 않은 환경(로컬 개발 등)
            answer = (
                "검색 데이터 또는 FAISS 인덱스가 현재 실행 환경에 연결되어 있지 않아 "
                "추천 후보를 조회하지 못했습니다."
            )
        else:
            answer = "조건에 맞는 추천 결과를 찾지 못했습니다."

        return {
            "answer": answer,
            "games": [],
            "next_question": _build_next_question(category, group),
        }

    # ── 아이템별 reason 조립 ──
    for item in items[:max_items]:
        title = item.get("title") or item.get("name") or "제목 없음"
        reasons: list[str] = []

        # 인원 조건 검사
        headcount = group.get("headcount")
        min_players = item.get("min_players")
        max_players = item.get("max_players")

        if headcount and isinstance(min_players, (int, float)) and isinstance(max_players, (int, float)):
            reasons.append(f"{headcount}명이 플레이하기에 적합한 인원 범위입니다.")

        # 평점 정보 추가
        rating = item.get("avg_rating") or item.get("rating")
        if isinstance(rating, (int, float)):
            reasons.append(f"평점 지표가 {rating}로 확인됩니다.")

        # 난이도(weight) 매칭 확인 — boardgame 전용
        weight_pref = group.get("weight_pref")
        weight = item.get("weight")
        if category == "boardgame" and isinstance(weight, (int, float)) and weight_pref:
            if weight_pref == "light" and weight < 2.5:
                reasons.append("입문자도 부담 없이 시작할 수 있는 난이도입니다.")
            elif weight_pref == "medium" and 2.5 <= weight <= 3.5:
                reasons.append("너무 가볍지도 무겁지도 않은 중간 난이도입니다.")
            elif weight_pref == "heavy" and weight > 3.5:
                reasons.append("전략적 깊이가 있는 고난도 게임입니다.")

        # 감정 태그 교집합 — 쿼리 태그와 아이템 태그가 겹치는 항목을 reason에 포함
        item_tags = set(item.get("emotion_tags", []) or [])
        query_tags = set(emotion_tags)
        matched_tags = sorted(item_tags & query_tags)

        if matched_tags:
            reasons.append(f"{', '.join(matched_tags)} 태그가 그룹 조건과 맞습니다.")

        # reason이 하나도 없으면 기본 문구로 대체
        if not reasons:
            reasons.append("검색 조건과 메타데이터 기준으로 상위에 노출된 추천 후보입니다.")

        games.append(
            {
                "title": title,
                "reason": " ".join(reasons),
                "matched_tags": matched_tags,
                "final_score": item.get("final_score") or item.get("total_score"),
                "emotion_tags": item.get("emotion_tags", []) or [],
                "source": item.get("source"),
                "avg_rating": item.get("avg_rating") or item.get("rating"),
                "min_players": item.get("min_players"),
                "max_players": item.get("max_players"),
                "image": item.get("image"),
            }
        )

    # ── 요약 answer 조립 ──
    headcount_text = f"{group.get('headcount')}명" if group.get("headcount") else "요청하신 조건"
    category_label = "보드게임" if category == "boardgame" else "머더미스터리"
    top_title = games[0]["title"] if games else ""

    answer = (
        f"{headcount_text} 그룹에 맞는 {category_label} 추천 결과입니다. "
        f"가장 우선 추천할 후보는 '{top_title}'입니다. "
        "인원 조건, 난이도, 태그 매칭, 평점 정보를 함께 고려했습니다."
    )

    return {
        "answer": answer,
        "games": games,
        "next_question": _build_next_question(category, group),
    }


# =====================================================================
# LangGraph Node 함수
# =====================================================================

def node_normalize_input(state: PipelineState) -> dict[str, Any]:
    """
    [노드: normalize_input]
    외부 입력 query/category를 내부 파이프라인 state로 변환한다.

    [수행 작업]
        1) query/user_text를 통합하여 단일 query 문자열로 정리
        2) category를 CATEGORY_ALIASES 기준으로 정규화
        3) query + 외부 group 병합하여 group dict 완성
        4) 이후 노드에서 사용할 state 필드를 초기값으로 세팅

    [설계 의도]
        입력 정규화를 가장 먼저 수행함으로써 이후 노드들이
        동일한 필드명/형식을 가정할 수 있게 한다.
    """

    query = state.get("query") or state.get("user_text") or ""
    query = str(query).strip()

    raw_category = state.get("category")
    category = _normalize_category(raw_category, query)

    group = _merge_group_from_query(
        query=query,
        category=category,
        group=state.get("group") or {},
    )

    return {
        "query": query,
        "user_text": query,
        "category": category,
        "group": group,
        "use_api": bool(state.get("use_api", True)),
        # ── 이후 노드에서 채워질 중간 state 초기화 ──
        "query_text": "",
        "query_filter": {},
        "emotion_tags": [],
        "anchor_titles": [],
        "retrieved_items": [],
        "filtered_items": [],
        "retrieve_error": "",
        "generate_error": "",
        "result": {},
        "answer": "",
        "games": [],
        "next_question": "",
    }


def node_check_sufficiency(state: PipelineState) -> dict[str, Any]:
    """
    [노드: check_sufficiency]
    RAG 검색을 실행하기 위한 최소 조건이 충족되었는지 판단한다.

    [최소 필수 조건]
        - query:    자연어 요청 존재
        - category: boardgame 또는 murdermystery
        - headcount: 인원 수 존재

    [설계 의도]
        이 세 가지 조건은 검색 결과의 유효성에 직결되므로 필수로 판단한다.
        나머지 조건(play_time, relation 등)은 선택 조건으로 보고,
        검색은 진행하되 generate 노드에서 next_question으로 보완한다.

    Returns:
        is_sufficient: True이면 query_transform으로 진행
        missing_fields: 부족한 필드명 목록 (clarify 노드에서 활용)
    """

    missing_fields: list[str] = []

    query = state.get("user_text") or state.get("query") or ""
    category = state.get("category") or ""
    group = state.get("group") or {}

    if not query:
        missing_fields.append("query")

    if category not in SUPPORTED_CATEGORIES:
        missing_fields.append("category")

    if not group.get("headcount"):
        missing_fields.append("headcount")

    return {
        "is_sufficient": len(missing_fields) == 0,
        "missing_fields": missing_fields,
    }


def route_after_sufficiency(state: PipelineState) -> str:
    """
    [라우터: check_sufficiency → clarify | query_transform]
    조건 충분 여부에 따라 다음 노드를 결정하는 조건부 엣지 함수.

    Returns:
        "query_transform": 최소 조건 충족 → RAG 파이프라인 진행
        "clarify":         조건 부족 → 역질문 반환 후 END
    """

    if state.get("is_sufficient"):
        return "query_transform"

    return "clarify"


def node_clarify(state: PipelineState) -> dict[str, Any]:
    """
    [노드: clarify]
    최소 조건이 부족한 경우 RAG 검색 없이 역질문만 반환한다.

    [흐름]
        check_sufficiency에서 missing_fields가 1개 이상인 경우 이 노드로 분기 →
        _build_next_question으로 가장 우선 보완 항목을 질문 →
        바로 END로 연결 (검색/생성 없음)
    """

    category = state.get("category") or ""
    group = state.get("group") or {}
    missing_fields = state.get("missing_fields") or []

    next_question = _build_next_question(
        category=category,
        group=group,
        missing_fields=missing_fields,
    )

    result = {
        "answer": "추천을 정확히 하기 위해 조건이 조금 더 필요합니다.",
        "games": [],
        "next_question": next_question,
    }

    return {
        "result": result,
        "answer": result["answer"],
        "games": result["games"],
        "next_question": result["next_question"],
    }


def node_query_transform(state: PipelineState) -> dict[str, Any]:
    """
    [노드: query_transform]
    그룹 조건 + 자연어 요청을 BM25/FAISS 검색용 구조화 쿼리로 변환한다.

    [위임]
        실제 변환 로직은 yoonha_query_transformer.transform()에 위임한다.

    [반환 필드]
        query_text:    BM25 검색용 자연어 텍스트
        query_filter:  하드 필터 조건 (players, max_time 등)
        emotion_tags:  감정 태그 목록 (tag_filter 노드에서 사용)
        anchor_titles: FAISS 앵커 타이틀 (retrieve 노드에서 사용)
    """

    from recommender.rag.yoonha_query_transformer import transform as query_transform

    transformed = query_transform(
        user_text=state.get("user_text", ""),
        group=state.get("group", {}),
        category=state.get("category", "boardgame"),
    )

    return {
        "query_text": transformed.get("query_text", ""),
        "query_filter": transformed.get("query_filter", {}),
        "emotion_tags": transformed.get("emotion_tags", []),
        "anchor_titles": transformed.get("anchor_titles", []),
    }


def node_retrieve(state: PipelineState) -> dict[str, Any]:
    """
    [노드: retrieve]
    BM25 + FAISS 하이브리드 검색을 실행하여 후보 아이템을 수집한다.

    [흐름]
        1) anchor_titles → get_embedding()으로 FAISS 검색 벡터 생성
        2) query_text + query_filter + query_vector → retrieve()로 하이브리드 검색
        3) topk=50개 반환

    [예외 처리]
        데이터/FAISS index가 로컬에 없으면 예외를 catch하고 빈 결과를 반환한다.
        retrieve_error에 예외 메시지를 기록하여 generate 노드에서 fallback 메시지 생성에 활용한다.
        실제 제출/실행 환경에서 data가 연결되어 있으면 정상 검색된다.
    """

    try:
        from recommender.rag.yoonha_hybrid_retriever import get_embedding, retrieve

        # anchor_titles → FAISS 검색 기준 벡터 생성
        query_vector = get_embedding(
            state.get("anchor_titles", []),
            state.get("category", "boardgame"),
        )

        items = retrieve(
            query_text=state.get("query_text", ""),
            query_filter=state.get("query_filter", {}),
            query_vector=query_vector,
            category=state.get("category", "boardgame"),
            topk=50,
        )

        return {
            "retrieved_items": items,
            "retrieve_error": "",
        }

    except Exception as exc:
        # 데이터 없는 환경 (로컬 개발, CI 등) 에서도 파이프라인이 깨지지 않도록 처리
        return {
            "retrieved_items": [],
            "retrieve_error": str(exc),
        }


def node_tag_filter(state: PipelineState) -> dict[str, Any]:
    """
    [노드: tag_filter]
    감정 태그 기반 필터링 및 점수 조정을 수행한다.

    [위임]
        실제 필터링/점수 조정은 yoonha_tag_filter.filter_and_score()에 위임한다.

    [horror_tolerance]
        group에서 horror_tolerance를 읽어 필터에 전달한다.
        값이 없으면 기본값 2(공포 가능)로 처리한다.

    [결과]
        filtered_items: 점수 조정 및 필터링이 완료된 최종 후보 리스트
    """

    items = state.get("retrieved_items", []) or []
    if not items:
        return {"filtered_items": []}

    from recommender.rag.yoonha_tag_filter import filter_and_score

    group = state.get("group", {})
    horror_tolerance = group.get("horror_tolerance", 2)   # None이면 공포 가능(2)으로 처리

    filtered = filter_and_score(
        items=items,
        emotion_tags=state.get("emotion_tags", []),
        horror_tolerance=horror_tolerance,
    )

    return {"filtered_items": filtered}


def node_generate(state: PipelineState) -> dict[str, Any]:
    """
    [노드: generate]
    추천 텍스트, 추천 게임 리스트, 역질문을 생성한다.

    [use_api 분기]
        use_api=True:
            yoonha_generator.generate() 호출 시도.
            성공하면 API 생성 결과 사용.
            실패하면 로컬 fallback + generate_error에 예외 기록.

        use_api=False:
            로컬 fallback(_generate_without_api_local)만 사용.
            OpenAI API 키 없이도 동작 가능.

    [공통]
        _normalize_result()로 출력을 항상 동일한 스펙으로 정규화한 후 반환.
    """

    items = state.get("filtered_items", []) or []
    group = state.get("group", {}) or {}
    category = state.get("category", "boardgame")
    emotion_tags = state.get("emotion_tags", []) or []
    retrieve_error = state.get("retrieve_error", "")

    use_api = bool(state.get("use_api", True))

    if use_api:
        try:
            from recommender.rag.yoonha_generator import generate

            result = generate(
                items=items,
                group=group,
                category=category,
                emotion_tags=emotion_tags,
            )

        except Exception as exc:
            # API 실패 → 로컬 fallback으로 대체하고 generate_error에 원인 기록
            result = _generate_without_api_local(
                items=items,
                group=group,
                category=category,
                emotion_tags=emotion_tags,
                retrieve_error=retrieve_error,
            )
            result["answer"] = result["answer"] + f" API 생성은 실패하여 룰 기반 결과로 대체했습니다."

            return {
                "result": _normalize_result(result),
                "answer": result.get("answer", ""),
                "games": result.get("games", []),
                "next_question": result.get("next_question", ""),
                "generate_error": str(exc),
            }

    else:
        result = _generate_without_api_local(
            items=items,
            group=group,
            category=category,
            emotion_tags=emotion_tags,
            retrieve_error=retrieve_error,
        )

    formatted = _normalize_result(result)

    return {
        "result": formatted,
        "answer": formatted["answer"],
        "games": formatted["games"],
        "next_question": formatted["next_question"],
        "generate_error": "",
    }


# =====================================================================
# Graph build
# =====================================================================

def build_graph():
    """
    LangGraph 노드·엣지를 구성하고 컴파일된 graph를 반환한다.

    [노드 등록 순서]
        normalize_input → check_sufficiency
            ├─ clarify → END
            └─ query_transform → retrieve → tag_filter → generate → END

    [input_schema / output_schema]
        LangGraph에 GraphInput / GraphOutput을 명시하여
        graph.invoke() 입출력 타입을 강제한다.
    """

    workflow = StateGraph(
        PipelineState,
        input_schema=GraphInput,
        output_schema=GraphOutput,
    )

    # ── 노드 등록 ──
    workflow.add_node("normalize_input", node_normalize_input)
    workflow.add_node("check_sufficiency", node_check_sufficiency)
    workflow.add_node("clarify", node_clarify)
    workflow.add_node("query_transform", node_query_transform)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("tag_filter", node_tag_filter)
    workflow.add_node("generate", node_generate)

    # ── 엔트리 포인트 ──
    workflow.set_entry_point("normalize_input")

    # ── 고정 엣지 ──
    workflow.add_edge("normalize_input", "check_sufficiency")

    # ── 조건부 엣지: 조건 충분 여부에 따라 분기 ──
    workflow.add_conditional_edges(
        "check_sufficiency",
        route_after_sufficiency,
        {
            "clarify": "clarify",
            "query_transform": "query_transform",
        },
    )

    # ── clarify 경로: 역질문 반환 후 바로 종료 ──
    workflow.add_edge("clarify", END)

    # ── RAG 경로: 검색 → 필터 → 생성 → 종료 ──
    workflow.add_edge("query_transform", "retrieve")
    workflow.add_edge("retrieve", "tag_filter")
    workflow.add_edge("tag_filter", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# =====================================================================
# 공개 인터페이스
# =====================================================================

# 모듈 임포트 시 1회 컴파일 — 이후 graph.invoke()로 재사용
graph = build_graph()


def run_pipeline(
    user_text: str,
    group: dict[str, Any] | None = None,
    category: str = "boardgame",
    use_api: bool = True,
) -> dict[str, Any]:
    """
    기존 yoonha_graph.py 호출 방식과의 하위 호환을 위한 래퍼 함수.

    [기존 호출 방식]
        run_pipeline(
            user_text="4명이서 할 전략 보드게임 추천해줘",
            group={"headcount": 4},
            category="boardgame",
            use_api=False,
        )

    [신규 권장 방식]
        graph.invoke({
            "query": "4명이서 할 보드게임",
            "category": "boardgame"
        })

    내부적으로 graph.invoke()를 호출하므로 동작은 동일하다.
    """

    return graph.invoke(
        {
            "query": user_text,
            "category": category,
            "group": group or {},
            "use_api": use_api,
        }
    )


# =====================================================================
# 로컬 테스트
# =====================================================================
if __name__ == "__main__":
    result = graph.invoke(
        {
            "query": "4명이서 할 보드게임",
            "category": "boardgame",
        }
    )

    print(result)