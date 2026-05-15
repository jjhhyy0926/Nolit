"""
recommender/rag/yoonha_tag_filter.py

[ 역할 ]
  런타임 감정 태깅 + 감정 태그 기반 필터링 + 점수 조정

[ 핵심 설계 ]
  원본 데이터 파일을 수정하지 않음.
  검색 결과(retrieved_items)에 런타임으로 감정 태그를 붙인 뒤,
  사용자 그룹 조건의 emotion_tags와 매칭하여 점수를 조정한다.

[ 태깅 차원 ] (기획서 기준)
  - 공포    : "생각보다 무서웠어요", "공포 없어서 좋았어요"
  - 난이도  : "너무 어려워서 막혔어요", "입문자도 쉽게 했어요"
  - 분위기  : "처음엔 어색했는데 금방 친해졌어요"
  - 인원조건: "4인으로 갔는데 딱 좋았어요"

[ 점수 계산 ]
  final_score = base_score + (positive_score + negative_score) × emotion_weight

[ 파이프라인 위치 ]
  retrieve 노드 → ★ tag_filter 노드 ★ → generate 노드
"""

from __future__ import annotations
import re

# ==========================================================
# 감정 태깅 (런타임 키워드 매칭)
# ==========================================================

# ── 공포 관련 키워드 ──
HORROR_HIGH_KEYWORDS = [
    "무서", "무섭", "공포", "호러", "깜짝", "점프스케어", "소름",
    "겁나", "겁났", "심장", "놀래", "놀랐", "어두", "으스스",
    "긴장", "떨렸", "비명", "잔인", "잔혹", "피", "시체",
    "살인", "귀신", "유령", "악몽",
]
HORROR_LOW_KEYWORDS = [
    "안 무서", "안무서", "무섭지 않", "무섭지않",
    "공포 없", "공포없", "공포가 없", "호러 없", "호러없",
    "겁쟁이도", "겁 많", "겁많", "무서운 거 없",
    "순한", "순둥", "귀여", "따뜻", "힐링", "평화",
]

# ── 난이도 관련 키워드 ──
DIFFICULTY_HIGH_KEYWORDS = [
    "어려", "어렵", "어려웠", "고난", "고난이도", "난이도 높",
    "막혔", "막힘", "진행이 안", "헤맸", "헤매", "복잡",
    "머리 아", "머리아", "두뇌", "빡세", "빡셌", "하드",
    "숙련자", "고수", "상급", "난해",
]
DIFFICULTY_LOW_KEYWORDS = [
    "쉬웠", "쉬움", "쉬운", "간단", "입문", "초보",
    "처음 해", "처음해", "입문용", "입문자",
    "금방", "빨리", "술술", "편하게", "부담 없", "부담없",
    "가볍", "가벼운", "룰이 쉬", "룰 쉬", "룰이 간단",
]

# ── 분위기 관련 키워드 ──
MOOD_POSITIVE_KEYWORDS = [
    "분위기 좋", "분위기좋", "재밌", "재미있", "즐거", "웃겼",
    "몰입", "빠져들", "몰입감", "완벽", "최고", "꿀잼",
    "수작", "명작", "역대급", "만족", "추천",
    "금방 친해", "금방친해", "어색함이 풀", "어색함이풀",
    "대화가 많", "대화가많", "토론", "떠들",
    "친해졌", "소통", "화기애애", "웃음",
]
MOOD_NEGATIVE_KEYWORDS = [
    "지루", "지겨", "노잼", "별로", "실망", "아쉬",
    "망했", "최악", "후회", "비추", "비추천",
    "정신없", "산만", "혼란", "난잡",
    "소외", "병풍", "할 게 없", "할게없", "뜬금",
]

# ── 인원 적합성 키워드 ──
PLAYER_FIT_KEYWORDS = [
    "딱 좋았", "딱좋았", "딱이었", "적절", "알맞",
    "인원 딱", "인원딱", "명이 딱", "명이딱",
    "풀인원", "최대인원", "최소인원",
]

# ── 관계 유형 키워드 ──
RELATION_KEYWORDS = {
    "처음만나는사이추천": [
        "처음 만나", "처음만나", "소개팅", "첫만남", "모르는 사",
        "어색한 사", "어색한사", "아이스브레이",
    ],
    "데이트추천": [
        "데이트", "커플", "연인", "둘이서", "2인", "이인",
        "로맨틱", "분위기 있",
    ],
    "친목용": [
        "친구", "친목", "모임", "회식", "단체", "워크샵",
        "팀빌딩", "동아리",
    ],
}


# ==========================================================
# 내부 헬퍼 함수
# ==========================================================

def _count_matches(text: str, keywords: list[str]) -> int:
    """텍스트에서 키워드 등장 횟수 합산. 대소문자 무시."""
    count = 0
    text_lower = text.lower()
    for kw in keywords:
        count += len(re.findall(re.escape(kw.lower()), text_lower))
    return count


def _extract_reviews_text(item: dict) -> str:
    """
    아이템에서 리뷰 텍스트를 추출.
    여러 필드를 순서대로 시도: reviews → review_text → description
    """
    reviews = item.get("reviews", "")
    if isinstance(reviews, list):
        return " ".join([str(r) for r in reviews])
    elif isinstance(reviews, str) and reviews:
        return reviews
    review_text = item.get("review_text", "")
    if review_text:
        return str(review_text)
    desc = item.get("description", "")
    return str(desc) if desc else ""


def _tag_from_text(text: str) -> list[str]:
    """
    리뷰 텍스트 기반 감정 태깅.
    키워드 등장 횟수를 세고, 임계값 기반으로 태그 부여.

    태깅 로직:
    - 공포: HIGH 3회 이상 + LOW의 2배 초과 → "공포있음" (8회 이상이면 "공포높음" 추가)
            LOW 2회 이상 + HIGH보다 많으면 → "공포없음"
            둘 다 없으면 → "공포없음" (공포 언급 자체가 없으면 비공포로 간주)
    - 난이도: HIGH/LOW 키워드 빈도 비교
    - 분위기: 긍정/부정 키워드 빈도 비교
    - 인원, 관계: 키워드 1회 이상 등장 시 태그 부여
    """
    tags = []

    # ── 공포 태깅 ──
    horror_high = _count_matches(text, HORROR_HIGH_KEYWORDS)
    horror_low = _count_matches(text, HORROR_LOW_KEYWORDS)
    if horror_high >= 3 and horror_high > horror_low * 2:
        tags.append("공포있음")
        if horror_high >= 8:
            tags.append("공포높음")
    elif horror_low >= 2 and horror_low > horror_high:
        tags.append("공포없음")
    elif horror_high == 0 and horror_low == 0:
        tags.append("공포없음")

    # ── 난이도 태깅 ──
    diff_high = _count_matches(text, DIFFICULTY_HIGH_KEYWORDS)
    diff_low = _count_matches(text, DIFFICULTY_LOW_KEYWORDS)
    if diff_high >= 3 and diff_high > diff_low * 2:
        tags.extend(["고난이도", "복잡함"])
    elif diff_low >= 2 and diff_low > diff_high:
        tags.extend(["입문용", "가볍게즐길수있음"])
    elif diff_low >= 1 and diff_high <= 1:
        tags.append("입문용")

    # ── 분위기 태깅 ──
    mood_pos = _count_matches(text, MOOD_POSITIVE_KEYWORDS)
    mood_neg = _count_matches(text, MOOD_NEGATIVE_KEYWORDS)
    if mood_pos >= 3 and mood_pos > mood_neg * 1.5:
        tags.append("분위기좋음")
        if _count_matches(text, ["웃", "웃겼", "웃음", "ㅋㅋ", "ㅎㅎ"]) >= 2:
            tags.append("웃음")
    elif mood_neg >= 3 and mood_neg > mood_pos:
        tags.append("분위기별로")

    # ── 대화유도 태깅 ──
    if _count_matches(text, ["대화", "토론", "소통", "떠들", "밀담"]) >= 2:
        tags.append("대화유도")

    # ── 인원 적합성 태깅 ──
    if _count_matches(text, PLAYER_FIT_KEYWORDS) >= 1:
        tags.append("인원적절")

    # ── 관계 유형 태깅 ──
    for tag_name, keywords in RELATION_KEYWORDS.items():
        if _count_matches(text, keywords) >= 1:
            tags.append(tag_name)

    return list(dict.fromkeys(tags))


def _tag_from_metadata(item: dict) -> list[str]:
    """
    리뷰 텍스트가 없거나 짧을 때 메타데이터 수치 기반으로 태깅.
    텍스트 태깅의 폴백(fallback) 역할.

    수치 기준:
        horror ≥ 3.5 → "공포있음" + "공포높음"
        horror ≥ 2.0 → "공포있음"
        horror < 1.0 → "공포없음"
        difficulty ≥ 4.0 → "고난이도" + "복잡함"
        difficulty ≤ 2.0 → "입문용" + "가볍게즐길수있음"
        weight ≥ 3.5 → "고난이도" (BGG weight 기준)
        satisfaction ≥ 4.0 → "분위기좋음"
        BGG rating ≥ 8.0 또는 기타 소스 rating ≥ 4.0 → "분위기좋음"
    """
    tags = []

    horror = item.get("horror")
    if isinstance(horror, (int, float)):
        if horror >= 3.5:
            tags.extend(["공포있음", "공포높음"])
        elif horror >= 2.0:
            tags.append("공포있음")
        elif horror < 1.0:
            tags.append("공포없음")

    difficulty = item.get("difficulty")
    if isinstance(difficulty, (int, float)):
        if difficulty >= 4.0:
            tags.extend(["고난이도", "복잡함"])
        elif difficulty <= 2.0:
            tags.extend(["입문용", "가볍게즐길수있음"])

    weight = item.get("weight")
    if isinstance(weight, (int, float)):
        if weight >= 3.5:
            tags.extend(["고난이도", "복잡함"])
        elif weight <= 2.0:
            tags.extend(["입문용", "가볍게즐길수있음"])

    satisfaction = item.get("satisfaction")
    if isinstance(satisfaction, (int, float)) and satisfaction >= 4.0:
        tags.append("분위기좋음")

    # ※ BGG는 10점 만점, 다른 소스는 5점 만점 → 서로 다른 기준 적용
    rating = item.get("avg_rating") or item.get("rating")
    if isinstance(rating, (int, float)):
        if (item.get("source") == "bgg" and rating >= 8.0) or \
                (item.get("source") != "bgg" and rating >= 4.0):
            tags.append("분위기좋음")

    return list(dict.fromkeys(tags))


def tag_item(item: dict) -> list[str]:
    """
    단일 아이템에 감정 태그 부여.
    1순위: 리뷰 텍스트 키워드 매칭
    2순위: 메타데이터 수치 기반 (리뷰 없을 때)
    """
    text = _extract_reviews_text(item)
    if text and len(text) > 20:
        return _tag_from_text(text)
    return _tag_from_metadata(item)


def _tag_items_runtime(items: list[dict]) -> None:
    """
    아이템 리스트에 emotion_tags 필드를 런타임으로 추가.
    이미 태그가 있는 아이템은 스킵 (idempotent).
    원본 데이터 파일은 수정하지 않음 — 메모리에서만 태그 부여.
    """
    for item in items:
        if not item.get("emotion_tags"):
            item["emotion_tags"] = tag_item(item)


# ==========================================================
# 감정 태그 필터링 + 점수 조정
# ==========================================================

# 공포 관련 태그 집합 (하드 필터용)
HORROR_TAGS = {
    "공포", "공포있음", "무서움", "호러", "공포요소",
    "깜짝놀람", "점프스케어", "어두움", "긴장감높음",
}
HORROR_STRONG_TAGS = {
    "공포높음", "극공포", "호러강함", "점프스케어",
}

# ── 긍정 태그 점수 (매칭 시 보너스) ──
POSITIVE_TAG_SCORE = {
    "입문용": 3,
    "분위기좋음": 3,
    "친목용": 3,
    "가족추천": 2,
    "데이트추천": 2,
    "공포없음": 2,
    "협력": 2,
    "웃음": 2,
    "가볍게즐길수있음": 2,
    "처음만나는사이추천": 3,
    "짧고간단": 1,
    "대화유도": 2,
}

# ── 부정 태그 점수 (아이템에 있으면 페널티) ──
NEGATIVE_TAG_SCORE = {
    "고난이도": -2,
    "복잡함": -2,
    "룰설명길다": -2,
    "공포있음": -3,
    "무서움": -3,
    "어두움": -1,
    "체력필요": -1,
    "긴장감높음": -2,
}


def _is_horror_blocked(item: dict, horror_tolerance: int) -> bool:
    """
    공포 하드 필터: 공포 수용도에 따라 아이템 차단 여부 결정.

    horror_tolerance:
        0 (공포 불가) → HORROR_TAGS 중 하나라도 있으면 차단
        1 (약간 가능) → HORROR_STRONG_TAGS(극공포)만 차단
        2 (가능)      → 차단 안 함
    """
    if horror_tolerance >= 2:
        return False
    item_tags = set(item.get("emotion_tags") or [])
    if horror_tolerance == 0:
        return bool(item_tags & HORROR_TAGS)
    else:
        return bool(item_tags & HORROR_STRONG_TAGS)


def _emotion_score(item: dict, emotion_tags: list[str]) -> float:
    """
    사용자 요청 감정 태그와 아이템 태그의 매칭 점수 계산 (양수).
    매칭된 태그마다 POSITIVE_TAG_SCORE에서 점수를 가져옴.
    """
    if not emotion_tags:
        return 0.0
    item_tags = set(item.get("emotion_tags") or [])
    score = 0.0
    for tag in emotion_tags:
        if tag in item_tags:
            score += POSITIVE_TAG_SCORE.get(tag, 1)
    return score


def _negative_score(item: dict) -> float:
    """
    아이템이 가진 부정 태그의 페널티 점수 합산 (음수).
    사용자 요청과 무관하게, 아이템 자체의 부정적 속성에 대한 감점.
    """
    item_tags = set(item.get("emotion_tags") or [])
    score = 0.0
    for tag in item_tags:
        score += NEGATIVE_TAG_SCORE.get(tag, 0)
    return score


# ====================================================
# 공개 인터페이스
# ====================================================
def filter_and_score(
        items: list[dict],
        emotion_tags: list[str],
        horror_tolerance: int = 2,
        emotion_weight: float = 5.0,
) -> list[dict]:
    """
    런타임 감정 태깅 + 공포 필터링 + final_score 계산.

    처리 흐름:
      0. 각 아이템에 런타임 감정 태그 부여 (텍스트/메타 기반)
      1. 공포 하드 필터: horror_tolerance에 따라 공포 아이템 제거
      2. 감정 태그 점수: 매칭 보너스 + 부정 페널티
      3. final_score = base_score + (보너스 + 페널티) × emotion_weight
      4. final_score 내림차순 정렬

    Args:
        items:             hybrid_retriever 출력 아이템 리스트
        emotion_tags:      사용자 그룹 조건에서 생성된 원하는 감정 태그
        horror_tolerance:  공포 수용도 0=불가, 1=약간, 2=가능
        emotion_weight:    감정 태그 점수에 곱할 가중치 (기본 5.0)

    Returns:
        필터링 + 점수 조정된 아이템 리스트 (final_score 내림차순)
        원본 아이템은 수정하지 않고 .copy()로 복사하여 반환
    """
    # 0. 런타임 감정 태깅
    _tag_items_runtime(items)

    result = []

    for item in items:
        # 1. 공포 하드 필터
        if _is_horror_blocked(item, horror_tolerance):
            continue

        # 2. 감정 태그 점수 계산
        pos_score = _emotion_score(item, emotion_tags)
        neg_score = _negative_score(item)
        emotion_adjustment = (pos_score + neg_score) * emotion_weight

        # 3. final_score 계산
        base_score = item.get("total_score") or item.get("avg_rating") or 0
        final_score = base_score + emotion_adjustment

        item_copy = item.copy()
        item_copy["emotion_match_score"] = round(pos_score, 2)
        item_copy["negative_score"] = round(neg_score, 2)
        item_copy["final_score"] = round(final_score, 2)
        result.append(item_copy)

    # 4. final_score 내림차순 정렬
    result.sort(key=lambda x: x["final_score"], reverse=True)
    return result


def get_matched_tags(item: dict, emotion_tags: list[str]) -> list[str]:
    """아이템과 요청 태그 중 실제 매칭된 태그 목록 반환."""
    item_tags = set(item.get("emotion_tags") or [])
    return [tag for tag in emotion_tags if tag in item_tags]
