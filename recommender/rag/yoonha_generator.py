"""
yoonha_generator.py
검색 결과 → LLM 추천 이유 + 역질문 생성

====================================================================
[역할]
    RAG 파이프라인의 마지막 단계.
    tag_filter를 거친 검색 결과 상위 N개를 받아서:
    1) 그룹 조건 + 아이템 정보를 LLM 프롬프트로 구성
    2) OpenAI API(gpt-4o-mini)로 추천 이유 텍스트 생성
    3) JSON 형식으로 파싱하여 반환

[설계 문서와의 대응]
    - 기획서 §3-2: "이게 좋습니다"가 아니라 "비슷한 조건의 그룹에서
      이런 경험이 있었습니다"를 근거로 제시
    - 기획서 §3-2: 실패 방지 중심 의사 결정 — 감정 태그로 리스크 근거 활용
    - 기획서 §4: 출력 구조 3가지 = 추천 결과 + 추천 이유 + 역질문
    - 필터 문서 §공통 주의사항 §3: weight/horror/difficulty 설명 필수
      → SYSTEM_PROMPT에 각 필드의 범위와 의미를 명시

[출력 스펙]
    {
        "answer": str,          # 추천 요약 (3~5문장)
        "games": [              # 개별 추천 항목
            {
                "title": str,
                "reason": str,       # 그룹 조건과 연결된 추천 이유
                "matched_tags": [],  # 매칭된 감정 태그
                "final_score": float,
                "source": str,
                ...
            }
        ],
        "next_question": str    # 역질문 (Clarifying Question)
    }

[이중 안전장치]
    1) generate()       — OpenAI API 사용 (기본)
    2) generate_without_api() — 룰 기반 fallback (API 장애 또는 테스트 시)
    graph.py의 node_generate()에서 use_api 플래그로 분기

의존:
    pip install openai python-dotenv
    .env 파일에 OPENAI_API_KEY 설정 필요
====================================================================
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────
# 환경 설정
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")                          # 프로젝트 루트의 .env에서 API 키 로드

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"                                   # 비용 효율 + 속도 고려한 모델 선택


# ─────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
당신은 오프라인 활동(보드게임, 머더미스터리) 추천 전문가입니다.

역할:
1. 검색 결과를 바탕으로 그룹에게 맞는 추천을 자연스러운 텍스트로 설명합니다.
2. 각 추천 항목의 추천 이유를 그룹 조건과 연결해서 설명합니다.
3. 감정 태그가 매칭된 경우 그 태그를 근거로 활용합니다.
4. 더 좋은 추천을 위한 역질문 1개를 합니다.

규칙:
- 한국어로 답변하세요.
- 답변은 JSON 형식으로 반환하세요.

weight 필드 설명 (보드게임):
- 0~5 범위, 높을수록 복잡하고 어려운 게임
- light: 2.5 미만, medium: 2.5~3.5, heavy: 3.5 초과

horror 필드 설명 (방탈출):
- 0~5 범위, 높을수록 공포 요소 강함. 0이면 공포 없음

difficulty 필드 설명 (머더미스터리):
- 머미나우: 1/2/3/4 이산형 (1=쉬움, 4=매우 어려움)
"""
# [필터 문서 §공통 주의사항 §3]
# weight/horror/difficulty 설명을 프롬프트에 반드시 명시해야
# LLM이 반대로 해석하는 것을 방지할 수 있다.


RESPONSE_FORMAT_INSTRUCTION = """\
반드시 아래 JSON 형식으로만 답변하세요. 다른 텍스트나 마크다운 없이 순수 JSON만 반환하세요.

{
  "answer": "그룹 조건을 고려한 전체 추천 요약 텍스트 (3~5문장)",
  "games": [
    {
      "title": "게임 제목",
      "reason": "이 그룹에 추천하는 이유 (2~3문장)",
      "matched_tags": ["매칭된 감정 태그"]
    }
  ],
  "next_question": "더 좋은 추천을 위한 역질문"
}
"""
# JSON 전용 출력을 강제하여 파싱 실패 가능성을 낮춘다.


# ─────────────────────────────────────────────
# 컨텍스트 빌더
# ─────────────────────────────────────────────
def _build_context(
    items: list[dict],
    group: dict,
    category: str,
    emotion_tags: list[str],
    max_items: int = 5,
) -> str:
    """
    LLM에 전달할 컨텍스트 문자열 생성.

    [구성]
        1) 그룹 조건 요약 — 인원, 시간, 난이도, 공포 수용도, 관계, 감정 태그
        2) 카테고리
        3) 추천 후보 상위 N개 — 타이틀, 평점, 인원, 시간, 카테고리, 소스, 감정 태그, 점수

    [설계 의도]
        LLM이 "왜 이 그룹에 이 게임이 맞는가"를 설명할 수 있도록
        그룹 조건과 아이템 메타데이터를 구조화된 텍스트로 제공한다.
    """
    lines = []

    # ── 그룹 조건 요약 ──
    lines.append("## 그룹 조건")
    lines.append(f"- 인원: {group.get('headcount', '미정')}명")
    if group.get("play_time"):
        lines.append(f"- 플레이 시간: {group['play_time']}분 이내")
    if group.get("weight_pref"):
        lines.append(f"- 난이도 선호: {group['weight_pref']}")
    if group.get("horror_tolerance") is not None:
        labels = {0: "공포 불가", 1: "약간 가능", 2: "가능"}
        lines.append(f"- 공포 수용도: {labels.get(group['horror_tolerance'], '미정')}")
    if group.get("relation"):
        lines.append(f"- 관계: {group['relation']}")
    if group.get("category"):
        lines.append(f"- 카테고리: {group['category']}")
    if group.get("mechanism"):
        lines.append(f"- 메커니즘: {group['mechanism']}")

    if emotion_tags:
        lines.append(f"- 감정 태그: {', '.join(emotion_tags)}")

    lines.append(f"\n## 카테고리: {category}")

    # ── 추천 후보 ──
    lines.append(f"\n## 추천 후보 (상위 {min(max_items, len(items))}개)")
    for i, item in enumerate(items[:max_items], 1):
        title = item.get("title", item.get("name", "?"))
        lines.append(f"\n### {i}. {title}")

        # 카테고리별로 표시할 메타데이터가 다름
        if category == "boardgame":
            lines.append(f"  - 평점: {item.get('avg_rating', '?')}")
            lines.append(f"  - 난이도(weight): {item.get('weight', '?')}")
            lines.append(f"  - 인원: {item.get('min_players', '?')}~{item.get('max_players', '?')}명")
            # 보드라이프는 min_time~max_time 범위형, BGG는 playing_time 단일값
            if item.get("source") == "boardlife":
                lines.append(f"  - 시간: {item.get('min_time', '?')}~{item.get('max_time', '?')}분")
            else:
                lines.append(f"  - 시간: {item.get('playing_time', '?')}분")
            cat = item.get("category", "")
            if isinstance(cat, list):
                cat = ", ".join(cat)
            lines.append(f"  - 카테고리: {cat}")
            lines.append(f"  - 소스: {item.get('source', '?')}")

        elif category == "murdermystery":
            lines.append(f"  - 평점: {item.get('rating', '?')}")
            lines.append(f"  - 인원: {item.get('min_players', '?')}~{item.get('max_players', '?')}명")
            if item.get("play_time"):
                lines.append(f"  - 시간: {item['play_time']}분")
            if item.get("description"):
                desc = str(item["description"])[:200]      # 설명 200자 제한 (토큰 절약)
                lines.append(f"  - 설명: {desc}")
            lines.append(f"  - 소스: {item.get('source', '?')}")

        # 감정 태그 (tag_filter에서 부여된 태그)
        item_tags = item.get("emotion_tags", [])
        if item_tags:
            lines.append(f"  - 감정 태그: {', '.join(item_tags)}")

        # 최종 점수 (디버깅 + LLM 참고용)
        if item.get("final_score"):
            lines.append(f"  - 최종 점수: {item['final_score']}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 공개 인터페이스 ① — OpenAI API 사용
# ─────────────────────────────────────────────
def generate(
    items: list[dict],
    group: dict,
    category: str,
    emotion_tags: list[str] | None = None,
    max_items: int = 5,
    temperature: float = 0.7,
) -> dict:
    """
    검색 결과를 바탕으로 LLM 추천 생성 (OpenAI API).

    [호출 흐름]
        graph.py → node_generate(use_api=True) → 여기

    [처리 과정]
        1) _build_context()로 프롬프트 구성
        2) gpt-4o-mini에 system + user 메시지 전송
        3) JSON 응답 파싱
        4) 원본 아이템 메타데이터를 games에 병합
        5) 파싱 실패 시 → raw_text를 answer에 넣는 graceful degradation
           API 에러 시 → 에러 메시지 반환

    Returns:
        {"answer": str, "games": list[dict], "next_question": str}
    """
    # 검색 결과가 없는 경우 → 즉시 안내 반환
    if not items:
        return {
            "answer": "조건에 맞는 추천 결과를 찾지 못했습니다.",
            "games": [],
            "next_question": "어떤 종류의 활동을 찾고 계신가요? 보드게임, 머더미스터리 중 선택해주세요.",
        }

    # 컨텍스트 구성
    context = _build_context(
        items, group, category,
        emotion_tags or [],
        max_items=max_items,
    )

    # OpenAI API 호출
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{context}\n\n{RESPONSE_FORMAT_INSTRUCTION}",
        },
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,    # 0.7: 다양성과 일관성의 균형
            max_tokens=1500,            # 추천 5개 + 역질문에 충분한 토큰
        )
        raw_text = response.choices[0].message.content.strip()

        # JSON 파싱 — LLM이 ```json 으로 감쌀 수 있으므로 제거
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

    except json.JSONDecodeError:
        # JSON 파싱 실패 → raw_text를 그대로 answer에 사용 (graceful degradation)
        result = {
            "answer": raw_text[:500],
            "games": [
                {
                    "title": items[0].get("title", "?"),
                    "reason": raw_text[:300],
                    "matched_tags": [],
                }
            ],
            "next_question": "추천이 마음에 드셨나요? 더 원하시는 조건이 있으면 알려주세요.",
        }
    except Exception as e:
        # API 호출 자체 실패 (네트워크, 인증 등)
        result = {
            "answer": f"추천 생성 중 오류가 발생했습니다: {str(e)}",
            "games": [],
            "next_question": "다시 시도해주세요.",
        }

    # ── 원본 아이템 메타데이터를 games에 병합 ──
    # LLM이 생성한 title로 원본 아이템을 매칭하여
    # final_score, emotion_tags, source, 이미지 등을 추가
    for game in result.get("games", []):
        matched_item = next(
            (it for it in items[:max_items]
             if it.get("title") == game.get("title") or it.get("name") == game.get("title")),
            None,
        )
        if matched_item:
            game["final_score"] = matched_item.get("final_score") or matched_item.get("total_score")
            game["emotion_tags"] = matched_item.get("emotion_tags", [])
            game["source"] = matched_item.get("source")
            game["avg_rating"] = matched_item.get("avg_rating") or matched_item.get("rating")
            game["min_players"] = matched_item.get("min_players")
            game["max_players"] = matched_item.get("max_players")
            game["image"] = matched_item.get("image")

    return result


# ─────────────────────────────────────────────
# 공개 인터페이스 ② — 룰 기반 fallback (API 없이)
# ─────────────────────────────────────────────
def generate_without_api(
    items: list[dict],
    group: dict,
    category: str,
    emotion_tags: list[str] | None = None,
    max_items: int = 5,
) -> dict:
    """
    OpenAI API 없이 룰 기반으로 추천 이유 생성.

    [용도]
        - 테스트 환경에서 API 키 없이 파이프라인 검증
        - API 장애 시 fallback
        - graph.py의 use_api=False 모드

    [로직]
        각 아이템에 대해:
        1) 인원 매칭 → "N명이 플레이하기에 적합"
        2) 평점 → "평점 X.X으로 높은 평가"
        3) 난이도(weight_pref) 매칭 → light/medium/heavy 설명
        4) 감정 태그 교집합 → "태그가 그룹 조건에 부합"

    [역질문 우선순위]
        weight_pref 없으면 → 난이도 질문
        play_time 없으면 → 시간 질문
        relation 없으면 → 관계 질문
    """
    games = []

    for item in items[:max_items]:
        title = item.get("title", item.get("name", "?"))
        reasons = []

        # 인원 매칭 이유
        headcount = group.get("headcount")
        if headcount:
            min_p = item.get("min_players", 0)
            max_p = item.get("max_players", 0)
            if min_p and max_p:
                reasons.append(f"{headcount}명이 플레이하기에 적합한 인원 구성입니다 ({min_p}~{max_p}명).")

        # 평점 이유
        rating = item.get("avg_rating") or item.get("rating")
        if rating:
            reasons.append(f"평점 {rating}으로 높은 평가를 받고 있습니다.")

        # 난이도 매칭 이유 (보드게임 전용)
        weight = item.get("weight")
        weight_pref = group.get("weight_pref")
        if weight and weight_pref:
            if weight_pref == "light" and weight < 2.5:
                reasons.append("가볍게 즐길 수 있는 난이도입니다.")
            elif weight_pref == "heavy" and weight > 3.5:
                reasons.append("전략적 깊이가 있는 고난이도 게임입니다.")

        # 감정 태그 교집합 매칭
        item_tags = set(item.get("emotion_tags", []))
        query_tags = set(emotion_tags or [])
        matched = list(item_tags & query_tags)
        if matched:
            reasons.append(f"'{', '.join(matched)}' 태그가 매칭되어 그룹 조건에 부합합니다.")

        reason = " ".join(reasons) if reasons else "추천 조건에 부합하는 항목입니다."

        games.append({
            "title": title,
            "reason": reason,
            "matched_tags": matched,
            "final_score": item.get("final_score") or item.get("total_score"),
            "emotion_tags": item.get("emotion_tags", []),
            "source": item.get("source"),
            "avg_rating": item.get("avg_rating") or item.get("rating"),
            "min_players": item.get("min_players"),
            "max_players": item.get("max_players"),
            "image": item.get("image"),
        })

    # ── 요약 텍스트 생성 ──
    headcount = group.get("headcount", "?")
    if games:
        top_title = games[0]["title"]
        answer = f"{headcount}명 그룹에 맞는 추천 결과입니다. 1순위는 '{top_title}'입니다."
    else:
        answer = "조건에 맞는 추천 결과를 찾지 못했습니다."

    # ── 역질문 생성 (부족한 조건 우선) ──
    next_question = "추천이 마음에 드셨나요?"
    if not group.get("weight_pref"):
        next_question = "게임 난이도는 어느 정도가 좋으세요? (가벼운 / 보통 / 어려운)"
    elif not group.get("play_time"):
        next_question = "플레이 시간은 어느 정도를 생각하고 계세요?"
    elif not group.get("relation"):
        next_question = "함께 하시는 분들과의 관계가 어떻게 되나요? (친구, 연인, 직장동료, 첫만남)"

    return {
        "answer": answer,
        "games": games,
        "next_question": next_question,
    }