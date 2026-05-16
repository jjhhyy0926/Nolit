import json
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

AI_FLOW = [
    "안녕하세요! 그룹에 맞는 여가 활동을 추천해드리겠습니다.\n먼저, 몇 명이서 활동하실 예정인가요?",
    "좋습니다! 처음 만나는 사이인가요, 아니면 이미 친한 사이인가요?\n관계에 따라 추천이 달라집니다.",
    "공포 요소에 대해 어떻게 생각하시나요?\n· 모두 괜찮음\n· 일부 민감함\n· 전체적으로 피하고 싶음",
    "예산은 1인당 얼마 정도를 생각하시나요?",
    "마지막으로, 활동성은 어느 정도 원하시나요?\n· 조용한 활동 선호\n· 보통\n· 활발한 활동 선호",
]

RECOMMENDATIONS = [
    {
        "rank": 1, "title": "스파이폴", "category": "보드게임", "rating": 4.6,
        "reason": "4인 그룹, 처음 만나는 사이에서 어색함을 빨리 푸는 데 효과적입니다. 대화를 유도하는 메커니즘이 자연스러운 친목 형성을 돕습니다.",
        "evidence": "처음 만나는 4인 그룹 후기 78건 중 '금방 친해졌다' 언급 비율 82%",
        "risk": None,
    },
    {
        "rank": 2, "title": "저스트 원", "category": "보드게임", "rating": 4.4,
        "reason": "협력형 게임으로 경쟁 부담이 없고, 룰이 단순해서 누구나 쉽게 즐길 수 있습니다.",
        "evidence": "초면 그룹 만족도 평균 4.4/5, 플레이타임 20분으로 부담 없음",
        "risk": None,
    },
    {
        "rank": 3, "title": "더 게임 익스트림", "category": "보드게임", "rating": 4.2,
        "reason": "완전 협력형이라 승패 부담이 없고, 30분 내외로 가벼운 워밍업에 적합합니다.",
        "evidence": "공포 민감 그룹 선택 비율 높음, 난이도 적정 평가 91%",
        "risk": "너무 단순해서 2라운드 이상은 지루할 수 있음 (15% 언급)",
    },
]

DEFAULT_PERSONA = {
    "groupSize": "4", "relationship": "friends",
    "horrorTolerance": "low", "activityLevel": "moderate", "budget": "20000",
}

SOURCE_TO_CATEGORY = {
    "bgg": "보드게임",
    "boardlife": "보드게임",
    "bbabang": "보드게임",
    "murmynow": "머더미스터리",
    "murdermysterylog": "머더미스터리",
}


def _rag_to_recommendations(games):
    """RAG 파이프라인 games 리스트를 프론트엔드 recommendations 포맷으로 변환"""
    result = []
    for i, game in enumerate(games[:3], 1):
        rating = game.get("avg_rating") or game.get("rating") or 0
        matched = game.get("matched_tags") or game.get("emotion_tags") or []
        score = game.get("final_score")
        source = game.get("source", "")
        category = SOURCE_TO_CATEGORY.get(source, "보드게임")

        if matched:
            evidence = "감정 태그 매칭: " + ", ".join(matched)
        elif score:
            evidence = "최종 점수: " + str(round(float(score), 3))
        else:
            evidence = "RAG 검색 결과 기반 추천"

        result.append({
            "rank": i,
            "title": game.get("title", "?"),
            "category": category,
            "rating": round(float(rating), 1) if rating else 0,
            "reason": game.get("reason", ""),
            "evidence": evidence,
            "risk": None,
        })
    return result


def home(request):
    return render(request, "recommender/home.html", {"current_page": "home"})


def ai(request):
    return render(request, "recommender/ai.html", {
        "current_page": "ai",
        "ai_flow_json": json.dumps(AI_FLOW, ensure_ascii=False),
        "recommendations_json": json.dumps(RECOMMENDATIONS, ensure_ascii=False),
    })

def persona(request):
    if request.method == "POST":
        request.session["persona"] = {
            "groupSize": request.POST.get("groupSize", "4"),
            "relationship": request.POST.get("relationship", "friends"),
            "horrorTolerance": request.POST.get("horrorTolerance", "low"),
            "activityLevel": request.POST.get("activityLevel", "moderate"),
            "budget": request.POST.get("budget", "20000"),
        }
        return HttpResponseRedirect(reverse("recommender:persona") + "?saved=1")

    persona_data = request.session.get("persona", DEFAULT_PERSONA)
    saved = request.GET.get("saved") == "1"
    return render(request, "recommender/persona.html", {
        "current_page": "persona",
        "persona": persona_data,
        "saved": saved,
    })


@csrf_exempt
@require_POST
def chat_api(request):
    try:
        body = json.loads(request.body)
        step = int(body.get("step", 0))
        message = body.get("message", "").strip()
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "invalid request"}, status=400)

    # 사용자 응답을 세션에 누적
    collected = request.session.get("chat_messages", [])
    if message:
        collected.append(message)
    request.session["chat_messages"] = collected
    request.session.modified = True

    next_step = step + 1

    # 아직 질문이 남아 있으면 다음 질문 반환
    if next_step < len(AI_FLOW):
        return JsonResponse({
            "reply": AI_FLOW[next_step],
            "done": False,
            "step": next_step,
        })

    # 모든 질문 완료 -> RAG 파이프라인 호출
    try:
        from recommender.yoonha_graph import run_pipeline

        # 수집된 답변으로 자연어 쿼리 구성
        query = " ".join(collected) if collected else "보드게임 추천해줘"

        # 사용자 답변에서 카테고리 키워드 탐지
        full_text = query.lower()
        if any(kw in full_text for kw in ["머더", "미스터리", "방탈출", "탈출"]):
            category = "murdermystery"
        else:
            category = "boardgame"

        rag_result = run_pipeline(user_text=query, category=category, use_api=True)

        # 세션 초기화 (다음 대화를 위해)
        request.session["chat_messages"] = []
        request.session.modified = True

        games = rag_result.get("games", [])
        answer = rag_result.get("answer", "그룹 조건을 분석했습니다.")
        next_q = rag_result.get("next_question", "")

        reply = (answer + "\n\n" + next_q) if next_q else answer
        recommendations = _rag_to_recommendations(games) if games else RECOMMENDATIONS

    except Exception:
        # RAG 실패 시 기존 하드코딩 fallback
        reply = "그룹 조건을 분석했습니다.\n비슷한 조건의 그룹 데이터를 기반으로 세 가지 추천을 드립니다."
        recommendations = RECOMMENDATIONS

    return JsonResponse({
        "reply": reply,
        "done": True,
        "step": next_step,
        "recommendations": recommendations,
    })


from recommender.prompts import QUICK_REPLIES, RAG_FALLBACK_MESSAGE
from recommender.rag.jinseo_slot_extractor import (
    extract_slots,
    merge_slots,
    missing_slots,
    build_followup,
    slots_to_query,
    slots_to_persona_text,
    slots_to_group,
)

# 슬롯 세션 키
_SLOT_SESSION_KEY = "chat_slots"


@require_GET
def quick_reply_api(request):
    """
    현재 대화 step에 맞는 빠른 답변 버튼 목록 반환
    GET /api/quickreply/?step=0
    """
    try:
        step = int(request.GET.get("step", 0))
    except ValueError:
        return JsonResponse({"error": "invalid step"}, status=400)

    buttons = QUICK_REPLIES[step] if 0 <= step < len(QUICK_REPLIES) else []
    return JsonResponse({"step": step, "buttons": buttons})


@csrf_exempt
@require_POST
def smart_chat_api(request):
    """
    자유 문장 + 역질문 방식의 스마트 챗봇 API
    POST /api/smart-chat/
    body: {"message": "4명이서 친한 친구끼리, 무서운 거 괜찮아, 2만원"}

    동작 흐름:
    1) 메시지에서 슬롯 추출 (LLM)
    2) 세션에 저장된 기존 슬롯과 병합
    3) 빠진 슬롯이 있으면 → 역질문 반환 (done=False)
    4) 모든 슬롯이 채워지면 → RAG 실행 후 추천 반환 (done=True)
    """
    # ── 요청 파싱 ────────────────────────────────────────────
    try:
        body = json.loads(request.body)
        message = body.get("message", "").strip()
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "invalid request"}, status=400)

    if not message:
        return JsonResponse({"error": "message is required"}, status=400)

    # ── 슬롯 추출 & 병합 ─────────────────────────────────────
    existing_slots = request.session.get(_SLOT_SESSION_KEY, {})
    new_slots = extract_slots(message)
    merged = merge_slots(existing_slots, new_slots)

    request.session[_SLOT_SESSION_KEY] = merged
    request.session.modified = True

    # ── 빠진 슬롯 확인 ──────────────────────────────────────
    missing = missing_slots(merged)

    # 빠진 슬롯이 있으면 역질문
    if missing:
        followup = build_followup(missing)
        return JsonResponse({
            "done"           : False,
            "reply"          : followup,
            "slots"          : merged,
            "missing_slots"  : missing,
        })

    # ── 모든 슬롯 완성 → RAG 실행 ───────────────────────────
    persona_text = slots_to_persona_text(merged)

    try:
        from recommender.yoonha_graph import run_pipeline

        full_text = message.lower()
        if any(kw in full_text for kw in ["머더", "미스터리"]):
            category = "murdermystery"
        elif any(kw in full_text for kw in ["방탈출", "탈출"]):
            category = "escape"
        else:
            category = "boardgame"

        group = slots_to_group(merged)
        query = slots_to_query(merged)   # user_text 맥락용

        rag_result = run_pipeline(
            user_text=query,   # 자연어 맥락
            group=group,       # 슬롯 직접 전달 (재파싱 방지)
            category=category,
            use_api=True,
        )

        games = rag_result.get("games", [])
        answer = rag_result.get("answer", "그룹 조건을 분석했습니다.")
        next_q = rag_result.get("next_question", "")
        reply = (answer + "\n\n" + next_q) if next_q else answer
        recommendations = _rag_to_recommendations(games) if games else RECOMMENDATIONS

    except Exception:
        reply = RAG_FALLBACK_MESSAGE
        recommendations = RECOMMENDATIONS

    # ── 세션 초기화 (다음 대화를 위해) ──────────────────────
    request.session[_SLOT_SESSION_KEY] = {}
    request.session.modified = True

    return JsonResponse({
        "done" : True,
        "reply" : reply,
        "slots" : merged,
        "persona_summary" : persona_text,
        "recommendations": recommendations,
    })


@csrf_exempt
@require_POST
def reset_slots_api(request):
    """
    슬롯 세션 초기화 (대화 처음부터 다시 시작)
    POST /api/reset-slots/
    """
    request.session[_SLOT_SESSION_KEY] = {}
    request.session.modified = True
    return JsonResponse({"message": "슬롯 초기화 완료"})