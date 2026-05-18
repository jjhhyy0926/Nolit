# 역할 : 사용자의 자유 문장에서 추천에 필요한 슬롯을 추출하고, 빠진 슬롯에 대해 역질문을 생성한다.
#
# 슬롯 목록 :
#   person_count : 인원 수 예) 4
#   relationship : 관계 예) "친한" | "처음"
#   horror_tolerance : 공포 허용도 예) "모두" | "일부" | "없음"
#   budget : 1인당 예산(원) 예) 20000
#   activity_level : 활동성 예) "조용" | "보통" | "활발"

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── 슬롯 스키마 ───────────────────────────────────────────────
SLOT_KEYS = [
    "person_count",
    "relationship",
    "horror_tolerance",
    "budget",
    "activity_level",
]

# 빠진 슬롯 → 역질문 매핑
FOLLOWUP_QUESTIONS = {
    "person_count" : "몇 명이서 활동하실 예정인가요?",
    "relationship" : "처음 만나는 사이인가요, 아니면 이미 친한 사이인가요?",
    "horror_tolerance" : "공포 요소에 대해 어떻게 생각하시나요?\n· 모두 괜찮음\n· 일부 민감함\n· 전체적으로 피하고 싶음",
    "budget" : "예산은 1인당 얼마 정도를 생각하시나요?",
    "activity_level" : "활동성은 어느 정도 원하시나요?\n· 조용한 활동 선호\n· 보통\n· 활발한 활동 선호",
}

# ── LLM 슬롯 추출 프롬프트 ────────────────────────────────────
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    ("system", """사용자 메시지에서 아래 5가지 슬롯을 추출해 JSON으로만 반환하세요.
값이 언급되지 않으면 null로 표시하세요. 설명은 생략하세요.

[상황 힌트]
현재 사용자는 다음 정보들에 대한 질문을 받았을 가능성이 높습니다: {missing_context}
사용자의 답변에 주어가 생략되어 있더라도, 위 힌트를 바탕으로 문맥을 유추해서 슬롯을 채워주세요.
(예: 현재 힌트에 'horror_tolerance'가 있고 사용자가 "피하고 싶어"라고 했다면, 공포를 피하고 싶다는 뜻이므로 "없음"으로 처리하세요.)

슬롯 정의:
- person_count : 정수 (예: 4)
- relationship : "친한" 또는 "처음"
- horror_tolerance : "모두" | "일부" | "없음" (공포를 피하고 싶다는 표현은 "없음"으로 매핑)
- budget : 정수, 1인당 원 단위 (예: 20000)
- activity_level : "조용" | "보통" | "활발"

출력 형식 (이것만 반환):
{{
    "person_count" : null,
    "relationship" : null,
    "horror_tolerance" : null,
    "budget" : null,
    "activity_level" : null
}}"""),
    ("human", "{message}"),
])

_chain = _prompt | _llm

# [수정된 부분] 매개변수에 missing 리스트를 추가로 받아서 프롬프트에 넣어줘!
def extract_slots(message: str, missing: list[str] = None) -> dict:
    """
    자유 문장 → 슬롯 딕셔너리 추출
    """
    # 빠진 슬롯을 문자열로 만들어 프롬프트에 전달 (없으면 '없음')
    missing_context = ", ".join(missing) if missing else "없음"
    
    # invoke할 때 missing_context도 같이 넣어준다!
    raw = _chain.invoke({
        "message": message, 
        "missing_context": missing_context
    }).content.strip()

    # 마크다운 코드블록 제거
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        slots = json.loads(raw)
    except json.JSONDecodeError:
        slots = {k: None for k in SLOT_KEYS}

    # 키 누락 보정
    for k in SLOT_KEYS:
        slots.setdefault(k, None)

    return slots


def merge_slots(existing: dict, new: dict) -> dict:
    """
    기존 세션 슬롯 + 새로 추출한 슬롯 병합
    새 값이 null이 아닌 경우에만 덮어쓴다.
    """
    merged = dict(existing)
    for k in SLOT_KEYS:
        if new.get(k) is not None:
            merged[k] = new[k]
    return merged


def missing_slots(slots: dict) -> list[str]:
    """null인 슬롯 키 목록 반환"""
    return [k for k in SLOT_KEYS if slots.get(k) is None]


def build_followup(missing: list[str]) -> str:
    """
    빠진 슬롯 중 첫 번째에 대한 역질문 문자열 반환
    (한 번에 하나씩 질문)
    """
    if not missing:
        return ""
    return FOLLOWUP_QUESTIONS[missing[0]]


def slots_to_query(slots: dict) -> str:
    """
    슬롯 딕셔너리 → RAG 파이프라인에 넘길 자연어 쿼리 생성
    """
    parts = []
    if slots.get("person_count"):
        parts.append(f"{slots['person_count']}명")
    if slots.get("relationship"):
        parts.append(slots["relationship"] + " 사이")
    if slots.get("horror_tolerance"):
        horror_map = {"모두": "공포 가능", "일부": "공포 일부 가능", "없음": "공포 없음"}
        parts.append(horror_map.get(slots["horror_tolerance"], ""))
    if slots.get("budget"):
        parts.append(f"예산 {slots['budget']}원")
    if slots.get("activity_level"):
        parts.append(f"활동성 {slots['activity_level']}")
    return " ".join(p for p in parts if p)


def slots_to_persona_text(slots: dict) -> str:
    """
    LLM 프롬프트에 넘길 그룹 조건 요약 텍스트 생성
    """
    lines = []
    if slots.get("person_count"):
        lines.append(f"인원: {slots['person_count']}명")
    if slots.get("relationship"):
        lines.append(f"관계: {slots['relationship']} 사이")
    if slots.get("horror_tolerance"):
        lines.append(f"공포 허용도: {slots['horror_tolerance']}")
    if slots.get("budget"):
        lines.append(f"예산: 1인당 {slots['budget']:,}원")
    if slots.get("activity_level"):
        lines.append(f"활동성: {slots['activity_level']}")
    return "\n".join(lines) if lines else "조건 없음"

def slots_to_group(slots: dict) -> dict:
    relation_map = {"친한": "friend", "처음": "first_meeting"}
    horror_map   = {"모두": 2, "일부": 1, "없음": 0}
    return {
        "headcount" : slots.get("person_count"),
        "relation" : relation_map.get(slots.get("relationship")),
        "horror_tolerance" : horror_map.get(slots.get("horror_tolerance")),
        "budget" : slots.get("budget"),
        "activity_level" : slots.get("activity_level"),
    }