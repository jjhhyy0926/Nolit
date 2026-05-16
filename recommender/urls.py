# ============================================================
# Django URL 라우팅 — 챗봇 페이지 + API 엔드포인트
# ============================================================
# 프로젝트 루트 urls.py에서 아래처럼 include:
#   path("", include("recommender.urls", namespace="recommender")),
# ============================================================

from django.urls import path

from . import views

app_name = 'recommender'

urlpatterns = [
    # 페이지
    path('', views.home, name='home'),                # 홈
    path('ai/', views.ai, name='ai'),                 # AI 추천 챗봇
    path('persona/', views.persona, name='persona'),  # 그룹 설정

    # API
    path('api/chat/', views.chat_api, name='chat_api'),                  # 챗봇 대화
    path("api/quickreply/", views.quick_reply_api, name="quick_reply"),  # 빠른 답변

    path('api/smart-chat/', views.smart_chat_api, name='smart_chat'),
    path('api/reset-slots/', views.reset_slots_api, name='reset_slots'),
]