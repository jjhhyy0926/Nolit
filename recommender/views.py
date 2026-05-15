import json
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

ACTIVITIES = [
    {
        "id": "1", "title": "윙스팬 (Wingspan)", "category": "boardgame",
        "rating": 4.6, "players": "1-5명", "time": "60분", "difficulty": "중급",
        "horror": None,
        "tags": ["엔진 빌딩", "카드 게임", "자연"],
        "description": "BGG 1위, 조류 테마의 전략 보드게임",
    },
    {
        "id": "2", "title": "비밀의 방", "category": "escape",
        "rating": 4.3, "players": "2-5명", "time": "70분", "difficulty": "중",
        "horror": "공포 없음",
        "tags": ["추리", "초보 추천", "홍대"],
        "description": "스토리 중심, 공포 요소 없는 입문자 추천 테마",
    },
    {
        "id": "3", "title": "한강", "category": "murder",
        "rating": 4.5, "players": "4-6명", "time": "120분", "difficulty": "중",
        "horror": None,
        "tags": ["추리", "협력", "입문용"],
        "description": "배우 참여형 머더미스터리, 초보자 추천",
    },
    {
        "id": "4", "title": "스파이폴", "category": "boardgame",
        "rating": 4.4, "players": "3-8명", "time": "15분", "difficulty": "입문",
        "horror": None,
        "tags": ["파티", "대화", "추리"],
        "description": "처음 만나는 사이에서 어색함을 빠르게 푸는 파티 게임",
    },
    {
        "id": "5", "title": "저스트 원", "category": "boardgame",
        "rating": 4.4, "players": "3-7명", "time": "20분", "difficulty": "입문",
        "horror": None,
        "tags": ["협력", "파티", "단어"],
        "description": "2019년 올해의 게임 수상작, 완전 협력형 파티 게임",
    },
    {
        "id": "6", "title": "공포의 저택", "category": "escape",
        "rating": 4.1, "players": "2-4명", "time": "60분", "difficulty": "중상",
        "horror": "공포 있음",
        "tags": ["공포", "액션", "강남"],
        "description": "공포 요소가 강한 액션형 방탈출 테마",
    },
]

CAT_EMOJI = {"escape": "🔐", "boardgame": "🎲", "murder": "🕵️", "crime": "🔍"}
CAT_LABEL = {"escape": "방탈출", "boardgame": "보드게임", "murder": "머더미스터리", "crime": "크라임씬"}

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


def explore(request):
    category = request.GET.get("category", "all")
    search = request.GET.get("search", "").strip().lower()
    difficulty = request.GET.get("difficulty", "")

    filtered = []
    for a in ACTIVITIES:
        if category != "all" and a["category"] != category:
            continue
        if search and search not in a["title"].lower() and not any(search in t.lower() for t in a["tags"]):
            continue
        if difficulty and a["difficulty"] != difficulty:
            continue
        a_copy = dict(a)
        a_copy["emoji"] = CAT_EMOJI.get(a["category"], "")
        a_copy["cat_label"] = CAT_LABEL.get(a["category"], "")
        filtered.append(a_copy)

    return render(request, "recommender/explore.html", {
        "current_page": "explore",
        "activities": filtered,
        "total": len(filtered),
        "sel_category": category,
        "sel_search": request.GET.get("search", ""),
        "sel_difficulty": difficulty,
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
