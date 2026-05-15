import streamlit as st
import html as _html

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="놀잇 NOLIT",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,400;0,600;0,700;0,800;0,900;1,700;1,800&family=Raleway:wght@400;500;600;700;900&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)
st.markdown("""
<style>

:root {
    --orange:     #FF6B00;
    --orange-lt:  #FF8C2A;
    --orange-dim: #E85F00;
    --teal:       #2BBFB0;
    --teal-lt:    #E6FAF8;
    --yellow:     #FFB800;
    --ink:        #2A2A2A;
    --ink-soft:   #4A4A4A;
    --muted:      #888888;
    --surface:    #F8F8F8;
    --border:     #E8E8E8;
    --border-em:  #CCCCCC;
    --tag-bg:     #FFF3E0;
    --radius:     10px;
    --radius-lg:  16px;
    --shadow-sm:  0 1px 4px rgba(0,0,0,0.06);
    --shadow-md:  0 4px 16px rgba(0,0,0,0.08);
    --shadow-or:  0 4px 20px rgba(255,107,0,0.20);
}
*, *::before, *::after { box-sizing: border-box; }
body, .stApp {
    background: #FFFFFF !important;
    color: var(--ink);
    font-family: 'Nunito', sans-serif;
}
.block-container { padding-top: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

/* 네비게이션 */
.topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 40px; height: 64px;
    background: #FFFFFF; border-bottom: 3px solid var(--orange);
    position: sticky; top: 0; z-index: 999;
}
.topnav-logo {
    font-family: 'Raleway', sans-serif;
    font-size: 1.1rem; font-weight: 900;
    color: var(--orange); letter-spacing: 0.1em;
    display: flex; align-items: center; gap: 10px;
}
.logo-pip {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--orange); display: inline-block;
}
.topnav-sub {
    font-size: 0.7rem; font-weight: 500; color: var(--muted);
    letter-spacing: 0.14em; text-transform: uppercase;
    border-left: 1px solid var(--border); padding-left: 12px; margin-left: 4px;
}

/* 버튼 */
.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important; border-radius: var(--radius) !important;
    transition: all 0.18s ease !important; letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"] {
    background: var(--orange) !important; color: #ffffff !important;
    border: 1.5px solid var(--border) !important; box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #FFFFFF !important; color: var(--teal) !important;
    border-color: var(--teal) !important; box-shadow: none !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important; color: var(--ink-soft) !important;
    border: 1.5px solid var(--border-em) !important;
    border-radius: var(--radius) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--teal-lt) !important; border-color: var(--teal) !important;
    color: #1A8A80 !important;
}

/* 카드 */
.card {
    background: #FFFFFF; border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 22px 24px; margin-bottom: 16px;
    box-shadow: var(--shadow-sm); transition: all 0.2s ease;
}
.card:hover {
    border-color: #E8E8E8; box-shadow: 0 6px 24px rgba(255,107,0,0.12);
    transform: translateY(-2px);
}

/* 배지 */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em; margin: 2px;
}
.badge-teal   { background: #E6FAF8; color: #1A8A80; border: 1px solid #7AD8D0; }
.badge-green  { background: #EAF5EC; color: #1C6B30; border: 1px solid #A8D8B0; }
.badge-red    { background: #FEF0EF; color: #9B2218; border: 1px solid #F0AAAA; }
.badge-yellow { background: #E6FAF8; color: #1A8A80; border: 1px solid #7AD8D0; }

/* 랭크 배지 */
.rank-gold   { background: var(--orange); color: #ffffff; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; letter-spacing: 0.05em; }
.rank-silver { background: #2BBFB0; color: #fff; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; }
.rank-bronze { background: #A0785A; color: #fff; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; }

/* 채팅 */
.chat-user {
    background: #ffffff; color: var(--ink);
    border: 1.5px solid var(--teal);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; margin: 8px 0;
    display: inline-block; max-width: 75%; font-size: 14px; line-height: 1.6;
}
.chat-ai {
    background: #FFF5F0; color: var(--ink);
    border: 1px solid #FFD0B0;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px; margin: 8px 0;
    display: inline-block; max-width: 85%; font-size: 14px; line-height: 1.6;
}
.chat-label {
    font-size: 10px; font-weight: 700; color: var(--teal);
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 5px;
}

/* 에비던스 */
.evidence-box {
    display: inline-block;
    background: #F0FCFB; border-left: 3px solid #7AD8D0;
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 8px 14px; font-size: 12.5px; color: #2A8A84;
    margin: 10px 0; line-height: 1.6;
}
.risk-box {
    background: #FEF9EE; border-left: 3px solid var(--yellow);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 8px 14px; font-size: 12.5px; color: #7A5000; margin: 8px 0;
}

/* 히어로 */
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--teal-lt); border: 1px solid var(--teal);
    border-radius: 20px; padding: 5px 16px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--teal); margin-bottom: 10px;
}
.hero-highlight {
    font-family: 'Nunito', sans-serif;
    font-weight: 900; font-style: italic; color: var(--orange);
}
hr { border: none; border-top: 1px solid var(--border); margin: 36px 0; }

/* 폼 */
.stTextInput > div > div > input {
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important; font-size: 14px !important;
    color: var(--ink) !important; background: #FFFFFF !important;
    padding: 10px 14px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(43,191,176,0.12) !important;
}
.stSelectbox > div > div {
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
h1, h2 {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 900 !important; color: var(--ink) !important;
    letter-spacing: -0.02em !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: var(--border-em); border-radius: 3px; }

/* Nunito 강제 전역 적용 */
* { font-family: 'Nunito', sans-serif !important; }
.topnav-logo, .topnav-logo * { font-family: 'Raleway', sans-serif !important; }
input, textarea, select, button,
[class*="st-"], [data-testid] * { font-family: 'Nunito', sans-serif !important; }

/* form_submit_button primary 색상 강제 - Streamlit 빨간색 override */
[data-testid="stFormSubmitButton"] button {
    background: #FF6B00 !important;
    border-color: #FF6B00 !important;
    color: #ffffff !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #E05F00 !important;
    border-color: #E05F00 !important;
    color: #ffffff !important;
}

/* 라디오 버튼 - 선택 텍스트 배경 제거 + 원 색상 주황 */
[data-testid="stRadio"] label { padding: 4px 0 !important; background: transparent !important; }
[data-testid="stRadio"] label span { background: transparent !important; color: var(--ink) !important; }
[data-testid="stRadio"] div[role="radiogroup"] { gap: 10px !important; }

/* 라디오 원 - 선택됨 */
[data-testid="stRadio"] input[type="radio"] { accent-color: #FF6B00 !important; }
[data-baseweb="radio"] [data-checked="true"] div { background: #FF6B00 !important; border-color: #FF6B00 !important; }
[data-baseweb="radio"] div { border-color: #CCCCCC !important; }
[data-baseweb="radio"] [data-checked="true"] div div { background: #ffffff !important; }

/* 라디오/셀렉트박스 간격 */
[data-testid="stRadio"] { margin-bottom: 8px !important; }
[data-testid="stSelectbox"] { margin-bottom: 8px !important; }

/* 페르소나 섹션 간격 */
.stRadio > div { gap: 8px !important; }

/* 파란 링크색 완전 제거 */
a { color: inherit !important; text-decoration: none !important; }
.stMarkdown a, .element-container a { color: inherit !important; }

/* 빠른선택 버튼 컬럼 gap 제거 */
[data-testid="column"] + [data-testid="column"] { gap: 0 !important; }
div[data-testid="stHorizontalBlock"]:has(.quick-row) { gap: 6px !important; }

/* selectbox label 높이 통일 - collapsed여도 공간 차지 방지 */
.stSelectbox label, .stTextInput label { display: none !important; }

/* 로고 링크 hover */
.topnav-logo { transition: opacity 0.15s; }
.topnav a:hover .topnav-logo { opacity: 0.75; }
.topnav a:hover .topnav-logo .logo-pip { background: var(--orange-dim); }

/* 폰트 통일 */
.stSelectbox, .stTextInput, .stRadio, .stMarkdown,
.stButton, div, span, p, label, input, select {
    font-family: 'Nunito', sans-serif !important;
}
h3, h4 { font-family: 'Nunito', sans-serif !important; font-weight: 700 !important; }
/* 로고 NOLIT → Raleway Black */
.topnav-logo { font-family: 'Raleway', sans-serif !important; font-weight: 900 !important; letter-spacing: 0.08em !important; }
.topnav-sub { font-family: 'Raleway', sans-serif !important; font-weight: 500 !important; }
/* nav 링크도 Raleway */
.stMarkdown a { font-family: 'Raleway', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data ─────────────────────────────────────────────────────────────────────
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
    ("assistant", "안녕하세요! 그룹에 맞는 여가 활동을 추천해드리겠습니다.\n먼저, 몇 명이서 활동하실 예정인가요?"),
    ("assistant", "좋습니다! 처음 만나는 사이인가요, 아니면 이미 친한 사이인가요?\n관계에 따라 추천이 달라집니다."),
    ("assistant", "공포 요소에 대해 어떻게 생각하시나요?\n· 모두 괜찮음\n· 일부 민감함\n· 전체적으로 피하고 싶음"),
    ("assistant", "예산은 1인당 얼마 정도를 생각하시나요?"),
    ("assistant", "마지막으로, 활동성은 어느 정도 원하시나요?\n· 조용한 활동 선호\n· 보통\n· 활발한 활동 선호"),
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

# ─── Session State ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [("assistant", AI_FLOW[0][1])]
if "chat_step" not in st.session_state:
    st.session_state.chat_step = 0
if "persona" not in st.session_state:
    st.session_state.persona = {
        "groupSize": "4", "relationship": "friends",
        "horrorTolerance": "low", "activityLevel": "moderate", "budget": "20000",
    }
if "persona_saved" not in st.session_state:
    st.session_state.persona_saved = False

# ─── Nav ───────────────────────────────────────────────────────────────────────
# 로고 클릭 → query_params로 홈 이동
if st.query_params.get("go") == "home":
    st.session_state.page = "home"
    st.query_params.clear()
    st.rerun()

st.markdown("""
<div class='topnav'>
    <a href='?go=home' target='_self' style='text-decoration:none'>
        <div class='topnav-logo'>
            NOLIT <span class='logo-pip'></span>
            <span class='topnav-sub'>놀잇 · AI 여가 추천</span>
        </div>
    </a>
</div>
""", unsafe_allow_html=True)

for _key in ["ai", "explore", "persona"]:
    if st.query_params.get("go") == _key:
        st.session_state.page = _key
        st.query_params.clear()
        st.rerun()

_page = st.session_state.page
_links = []
for _label, _key in [("AI 추천", "ai"), ("탐색하기", "explore"), ("설정", "persona")]:
    if _page == _key:
        _style = "color:#FF5200 !important;font-weight:700;border-bottom:2px solid #FF5200;padding-bottom:2px;"
    else:
        _style = "color:#888888 !important;font-weight:600;"
    _links.append(f"<a href='?go={_key}' target='_self' style='text-decoration:none;font-family:'Raleway',sans-serif;font-weight:600;font-size:14px;letter-spacing:0.04em;padding:4px 0;{_style}'>{_label}</a>")

_nav_html = "".join(_links)
st.markdown(
    "<div style='display:flex;justify-content:center;gap:36px;padding:14px 0 12px;'>"
    + _nav_html
    + "</div><hr style='margin:0;border-color:#EAE5DC;'>",
    unsafe_allow_html=True
)


# ─── HOME ─────────────────────────────────────────────────────────────────────
def page_home():
    # Hero
    st.markdown("""
    <div style='text-align:center; padding: 48px 0 36px'>
        <div class='hero-tag'>⚄ AI 기반 그룹 여가 의사결정 플랫폼</div>
        <h1 style='font-size:2.8rem; font-weight:800; line-height:1.3; margin-top:8px; color:#2A2A2A'>
            "이번 토요일 뭐 하지?"<br>
            이제 <span class='hero-highlight'>NOLIT이 답합니다</span>
        </h1>
        <p style='color:#7A7469; font-size:1.05rem; margin-top:18px; line-height:1.8'>
            그룹 조건을 이해하고 실패 없는 선택으로 안내하는<br>
            RAG 기반 여가 활동 추천 서비스
        </p>
    </div>
    """, unsafe_allow_html=True)



    st.markdown("<hr>", unsafe_allow_html=True)

    # 문제 정의
    st.markdown("""
    <h2 style='text-align:center; font-size:1.9rem; margin-bottom:10px'>
        왜 항상 <span class='hero-highlight'>"그냥 밥 먹자"</span>로 끝날까요?
    </h2>
    <p style='text-align:center; color:#7A7469; margin-bottom:36px'>
        콘텐츠는 충분합니다. 없는 건 <strong style='color:#4A4A4A'>우리 그룹에 맞는 선택</strong>입니다.
    </p>
    """, unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    for col, icon, title, desc in [
        (pc1, "🎭", "그룹 조건 미반영", "공포를 싫어하는 사람이 있는지, 처음 만나는 사이인지 고려되지 않습니다."),
        (pc2, "💸", "실패 비용이 큼", "방탈출 한 번 잘못 골라서 그날 분위기 전체가 망가집니다."),
        (pc3, "🗺️", "정보 파편화", "정보는 어딘가에 있지만 우리한테 맞는지 판단할 수 없습니다."),
    ]:
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center; padding:32px 24px'>
                <div style='font-size:2.2rem; margin-bottom:14px'>{icon}</div>
                <h4 style='margin:0 0 8px; font-size:0.95rem; font-weight:700;
                           color:#2A2A2A; letter-spacing:0.02em'>{title}</h4>
                <p style='color:#7A7469; font-size:13px; margin:0; line-height:1.7'>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 솔루션
    st.markdown("""
    <h2 style='text-align:center; font-size:1.9rem; margin-bottom:40px'>
        NOLIT 은 이렇게 해결합니다
    </h2>""", unsafe_allow_html=True)

    for num, icon, title, desc, example in [
        ("01", "💬", "AI 역질문으로 조건 파악",
         "막연하게 물어봐도 됩니다. AI가 관계, 공포 수용도, 예산 등 필요한 정보를 하나씩 끌어냅니다.",
         '"우리 4명인데 뭐 하면 좋을까?" → AI가 관계·공포 수용도·예산 등을 질문'),
        ("02", "📊", "RAG 기반 근거 있는 추천",
         "비슷한 조건의 그룹 경험 데이터를 근거로 제시합니다. 감이 아닌 데이터로 결정합니다.",
         '"4인 그룹에서 어색함이 빨리 풀렸다는 후기 비율 82%"'),
        ("03", "🛡️", "실패 방지 중심 설계",
         "좋은 선택보다 이 선택이 망할 수 있습니다를 먼저 알려줍니다.",
         '"공포 민감 인원 포함 시 만족도 2.1/5 → 대안 제시"'),
    ]:
        st.markdown(f"""
        <div class='card' style='display:flex; gap:20px; align-items:flex-start; padding:28px'>
            <div style='flex-shrink:0; width:40px; height:40px;
                        background:#FFF3E0; color:#FF6B00;
                        border-radius:10px; display:flex; align-items:center;
                        justify-content:center;
                        font-family:'Nunito', sans-serif;
                        font-weight:900; font-size:0.85rem;
                        letter-spacing:0.04em'>{num}</div>
            <div style='flex:1'>
                <h4 style='margin:0 0 6px; font-size:0.95rem; font-weight:700;
                           color:#2A2A2A'>{icon} {title}</h4>
                <p style='color:#7A7469; font-size:13px; margin-bottom:10px;
                          line-height:1.7'>{desc}</p>
                <div class='evidence-box'>{example}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 카테고리
    st.markdown("""
    <h2 style='text-align:center; font-size:1.9rem; margin-bottom:10px'>다양한 카테고리 지원</h2>
    <p style='text-align:center; color:#7A7469; margin-bottom:32px; font-size:0.95rem'>
        방탈출부터 머더미스터리까지, 실제 후기 데이터 기반 추천
    </p>""", unsafe_allow_html=True)

    cats_col = st.columns(4)
    for col, emoji, cat, desc, tag in [
        (cats_col[0], "🔐", "방탈출", "공포도·난이도·지역 기반 필터링", "어워즈 수상 가중치"),
        (cats_col[1], "🎲", "보드게임", "메커니즘·인원·복잡도 탐색", "BGG 데이터 연동"),
        (cats_col[2], "🕵️", "머더미스터리", "배우 참여형·대본형 유형별 분류", "라이트 태깅 RAG"),
        (cats_col[3], "🔍", "크라임씬", "난이도·분위기 기반 추천", "실제 후기 분석"),
    ]:
        with col:
            st.markdown(f"""
            <div class='card' style='padding:28px 20px'>
                <div style='font-size:2.2rem; margin-bottom:14px'>{emoji}</div>
                <h4 style='margin:0 0 6px; font-size:0.95rem; font-weight:700;
                           color:#2A2A2A'>{cat}</h4>
                <p style='color:#7A7469; font-size:12px; margin-bottom:12px;
                          line-height:1.6'>{desc}</p>
                <span class='badge badge-teal'>{tag}</span>
            </div>""", unsafe_allow_html=True)

    # CTA
    st.markdown("""
    <div style='border:1.5px solid #E8E8E8; border-radius:20px; padding:52px 40px;
                text-align:center; margin-top:40px; margin-bottom:48px;
                background:#FFFFFF; position:relative; overflow:hidden;'>
        <div style='position:absolute;top:-30px;right:-30px;width:160px;height:160px;
                    background:#FFF3E0;border-radius:50%;z-index:0'></div>
        <div style='position:absolute;bottom:-40px;left:-20px;width:120px;height:120px;
                    background:#E6FAF8;border-radius:50%;z-index:0'></div>
        <div style='position:relative;z-index:1'>
            <div style='display:inline-block;background:#E6FAF8;color:#1A8A80;
                        border-radius:20px;padding:4px 14px;font-size:11px;
                        font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                        margin-bottom:14px'>지금 바로 시작하세요</div>
            <div style='font-family:Nunito,sans-serif;
                        font-size:2rem; font-weight:900; color:#2A2A2A;
                        margin-bottom:10px; letter-spacing:-0.02em; line-height:1.25'>
                그룹에 딱 맞는 선택,<br>
                <span style='color:#FF6B00; font-weight:900'>AI가 찾아드립니다</span>
            </div>
            <p style='color:#888888; margin-bottom:28px; font-size:0.88rem; line-height:1.7'>
                방탈출 · 보드게임 · 머더미스터리 · 크라임씬
            </p>
            <a href='?go=ai' target='_self' style='display:inline-block;
                background:#FF6B00; color:#ffffff;
                padding:14px 36px; border-radius:10px;
                font-family:Nunito,sans-serif; font-weight:900;
                font-size:14px; text-decoration:none; letter-spacing:0.02em;
                box-shadow:0 4px 16px rgba(255,107,0,0.3)'>
                AI 추천 받기 →
            </a>
        </div>
    </div>""", unsafe_allow_html=True)


# ─── AI ───────────────────────────────────────────────────────────────────────
def page_ai():
    # 중앙 정렬 컨테이너
    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.markdown("""
        <div style='padding: 28px 0 12px'>
            <h2 style='font-size:1.4rem; margin-bottom:4px; color:#2A2A2A'>AI 추천</h2>
            <p style='color:#888; font-size:13px; margin:0'>
                AI가 역질문을 통해 그룹 조건을 파악하고, 실패 없는 선택을 추천합니다.
            </p>
        </div>""", unsafe_allow_html=True)

        # 대화 렌더링
        chat_html = ""
        for role, content in st.session_state.chat_messages:
            safe = _html.escape(content).replace("\n", "<br>")
            if role == "user":
                chat_html += f"""
                <div style='display:flex;justify-content:flex-end;margin:6px 0'>
                    <div class='chat-user'>{safe}</div>
                </div>"""
            elif content != "__RECOMMENDATIONS__":
                chat_html += f"""
                <div style='display:flex;align-items:flex-start;gap:8px;margin:6px 0'>
                    <div style='width:24px;height:24px;border-radius:50%;background:#E6FAF8;
                                display:flex;align-items:center;justify-content:center;
                                font-size:11px;flex-shrink:0;margin-top:4px;color:#2BBFB0'>✦</div>
                    <div>
                        <div class='chat-label'>NOLIT AI</div>
                        <div class='chat-ai'>{safe}</div>
                    </div>
                </div>"""

        chat_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
        body{margin:0;padding:0;font-family:'Nunito',sans-serif;}
        html,body{width:100%;margin:0;padding:0;} .wrap{background:#fff;border:1px solid #EEEEEE;border-radius:16px;padding:20px;min-height:280px;box-sizing:border-box;}
        .chat-label{font-size:9px;font-weight:800;color:#2BBFB0;letter-spacing:.16em;text-transform:uppercase;margin-bottom:4px;}
        .chat-ai{background:#F8F8F8;color:#2A2A2A;border-radius:4px 16px 16px 16px;padding:10px 16px;display:inline-block;max-width:92%;font-size:13.5px;line-height:1.65;}
        .chat-user{background:#fff;color:#2A2A2A;border:1.5px solid #2BBFB0;border-radius:16px 4px 16px 16px;padding:10px 16px;display:inline-block;max-width:92%;font-size:13.5px;line-height:1.65;}
        </style>
        """
        import streamlit.components.v1 as components
        components.html(chat_css + f"<div class='wrap'>{chat_html}</div>", height=380, scrolling=True)

        # 추천 결과
        if st.session_state.chat_step >= len(AI_FLOW):
            st.markdown("""
            <h3 style='font-weight:800;font-size:1.1rem;color:#2A2A2A;margin:20px 0 12px'>
                그룹 맞춤 추천 결과
            </h3>""", unsafe_allow_html=True)
            for rec in RECOMMENDATIONS:
                rank_cls = ["", "rank-gold", "rank-silver", "rank-bronze"][rec["rank"]]
                risk_html = f"<div class='risk-box'>⚠ {rec['risk']}</div>" if rec["risk"] else ""
                st.markdown(f"""
                <div class='card'>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:12px'>
                        <div style='display:flex; align-items:center; gap:10px'>
                            <span class='{rank_cls}'>#{rec['rank']}</span>
                            <strong style='font-size:1rem; color:#2A2A2A;
                                           font-family:'Nunito', sans-serif'>{rec['title']}</strong>
                            <span class='badge badge-teal'>{rec['category']}</span>
                        </div>
                        <span style='font-family:'Nunito', sans-serif;
                                     font-size:1.15rem; font-weight:700;
                                     color:#2A2A2A'>{rec['rating']}
                            <span style='color:#7A7469; font-size:11px; font-family:sans-serif'>/5</span>
                        </span>
                    </div>
                    <div style='display:flex; gap:10px; align-items:flex-start; margin-bottom:8px'>
                        <span style='color:#FF5200; font-size:16px; flex-shrink:0;
                                     margin-top:1px'>✓</span>
                        <p style='font-size:13px; margin:0; line-height:1.7;
                                  color:#4A4A4A'>{rec['reason']}</p>
                    </div>
                    <div class='evidence-box'>{rec['evidence']}</div>
                    {risk_html}
                </div>""", unsafe_allow_html=True)

        # 입력폼
        with st.form(key="chat_form", clear_on_submit=True):
            col_in, col_btn = st.columns([5, 1])
            with col_in:
                user_input = st.text_input("", placeholder="메시지를 입력하세요...",
                                           label_visibility="collapsed")
            with col_btn:
                send = st.form_submit_button("전송 →", type="primary", use_container_width=True)

        # 빠른 선택
        st.markdown("<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#aaa;margin:10px 0 6px'>빠른 선택</p>", unsafe_allow_html=True)
        quick = ["4명", "처음 만나는 사이", "공포 싫어하는 사람 있어요", "1인당 2만원"]
        q_html = "<div style='display:flex;gap:6px;flex-wrap:wrap'>"
        for q in quick:
            q_html += f"<a href='?quick={q}' target='_self' style='display:inline-block;padding:6px 14px;border:1px solid #DDD;border-radius:20px;font-size:12px;font-weight:600;color:#555;text-decoration:none;background:#fff;white-space:nowrap'>{q}</a>"
        q_html += "</div>"
        st.markdown(q_html, unsafe_allow_html=True)
        if st.query_params.get("quick"):
            _q = st.query_params.get("quick")
            st.query_params.clear()
            _process_input(_q)

        if send and user_input.strip():
            _process_input(user_input)



def _process_input(text: str):
    st.session_state.chat_messages.append(("user", text))
    step = st.session_state.chat_step + 1
    if step < len(AI_FLOW):
        st.session_state.chat_messages.append(("assistant", AI_FLOW[step][1]))
    else:
        st.session_state.chat_messages.append((
            "assistant",
            "그룹 조건을 분석했습니다.\n비슷한 조건의 그룹 데이터를 기반으로 세 가지 추천을 드립니다."
        ))
    st.session_state.chat_step = step
    st.rerun()


# ─── EXPLORE ──────────────────────────────────────────────────────────────────
def page_explore():
    st.markdown("""
    <div style='padding: 32px 0 16px'>
        <h2 style='font-size:1.6rem; margin-bottom:6px; color:#2A2A2A'>카테고리 탐색</h2>
        <p style='color:#7A7469; font-size:13.5px; margin:0'>
            다양한 여가 활동을 둘러보고 그룹에 맞는 선택을 찾아보세요.
        </p>
    </div>""", unsafe_allow_html=True)

    col_cat, col_search, col_diff = st.columns([2, 3, 2])
    with col_cat:
        cat_map = {"전체": "all", "🔐 방탈출": "escape", "🎲 보드게임": "boardgame",
                   "🕵️ 머더미스터리": "murder", "🔍 크라임씬": "crime"}
        sel_cat_label = st.selectbox("카테고리", list(cat_map.keys()), key="cat_filter",
                                     label_visibility="collapsed")
        sel_cat = cat_map[sel_cat_label]
    with col_search:
        search = st.text_input("검색", placeholder="🔎  제목이나 태그로 검색...",
                               label_visibility="collapsed")
    with col_diff:
        diff = st.selectbox("난이도", ["모든 난이도", "입문", "중급", "고급"], key="diff_filter",
                            label_visibility="collapsed")

    filtered = [
        a for a in ACTIVITIES
        if (sel_cat == "all" or a["category"] == sel_cat)
        and (not search or search.lower() in a["title"].lower()
             or any(search.lower() in t.lower() for t in a["tags"]))
        and (diff == "모든 난이도" or a["difficulty"] == diff)
    ]

    st.markdown(f"<p style='color:#7A7469; font-size:12px; margin-bottom:20px; font-weight:600; letter-spacing:0.06em; text-transform:uppercase'>{len(filtered)}개 결과</p>", unsafe_allow_html=True)

    if not filtered:
        st.info("검색 결과가 없습니다. 다른 조건으로 시도해보세요.")
        return

    cols = st.columns(3)
    for i, act in enumerate(filtered):
        with cols[i % 3]:
            horror_html = ""
            if act["horror"]:
                cls = "badge-red" if act["horror"] == "공포 있음" else "badge-green"
                horror_html = f"<span class='badge {cls}'>{act['horror']}</span>"
            tags_html = " ".join(f"<span class='badge badge-teal'>{t}</span>"
                                 for t in act["tags"])
            st.markdown(f"""
            <div class='card' style='padding:22px'>
                <div style='display:flex; justify-content:space-between;
                            align-items:flex-start; margin-bottom:10px'>
                    <div style='display:flex; align-items:center; gap:10px'>
                        <div style='width:40px; height:40px; background:#E6FAF8;
                                    border-radius:10px; display:flex; align-items:center;
                                    justify-content:center; font-size:1.3rem; flex-shrink:0'>
                            {CAT_EMOJI.get(act['category'], '')}
                        </div>
                        <div>
                            <strong style='font-size:0.9rem; color:#2A2A2A;
                                           font-weight:700; display:block'>{act['title']}</strong>
                            <span style='color:#7A7469; font-size:11px; font-weight:600;
                                         letter-spacing:0.06em; text-transform:uppercase'>
                                {CAT_LABEL.get(act['category'], '')}</span>
                        </div>
                    </div>
                    <span style='font-family:'Nunito', sans-serif;
                                 font-weight:700; font-size:1rem;
                                 color:#FF5200'>★ {act['rating']}</span>
                </div>
                <p style='color:#7A7469; font-size:12.5px; margin:0 0 12px;
                          line-height:1.65'>{act['description']}</p>
                <div style='display:grid; grid-template-columns:1fr 1fr;
                            gap:4px 8px; font-size:12px; color:#4A4A4A;
                            margin-bottom:12px'>
                    <span>👥 {act['players']}</span>
                    <span>⏱ {act['time']}</span>
                    <span>⚡ {act['difficulty']}</span>
                    <span>{horror_html}</span>
                </div>
                <div>{tags_html}</div>
            </div>""", unsafe_allow_html=True)


# ─── PERSONA ──────────────────────────────────────────────────────────────────
def page_persona():
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style='padding: 28px 0 12px'>
            <h2 style='font-size:1.4rem; margin-bottom:4px; color:#2A2A2A'>내 설정</h2>
            <p style='color:#888; font-size:13px; margin:0'>
                자주 사용하는 그룹 조건을 저장하면 AI 추천 시 자동으로 반영됩니다.
            </p>
        </div>""", unsafe_allow_html=True)

    with st.container():
        _, center, _ = st.columns([1, 2, 1])
        with center:


            size_map = {"2명 (데이트, 소개팅)": "2", "3명": "3", "4명": "4",
                        "5명": "5", "6명": "6", "8명+ (팀 모임)": "8"}
            rel_map = {"처음 만나는 사이": "first", "지인": "acquaintance",
                   "친한 친구": "friends", "가족": "family"}
            hor_map = {"모두 공포 괜찮음": "high",
                   "일부 민감함 (공포 제외 권장)": "low",
                   "전체적으로 피하고 싶음": "none"}
            act_map = {"조용한 활동 선호 (전략 보드게임, 추리)": "low",
                   "보통": "moderate",
                   "활발한 활동 선호 (액션 방탈출, 파티 게임)": "high"}
            bud_map = {"~1만원": "10000", "~2만원": "20000", "~3만원": "30000",
                   "~5만원": "50000", "5만원 이상": "100000"}

            p = st.session_state.persona

            st.markdown("<p style='font-size:11px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#7A7469; margin-bottom:8px'>주요 그룹 인원</p>", unsafe_allow_html=True)
            size_label = next((k for k, v in size_map.items() if v == p["groupSize"]), "4명")
            sel_size = st.selectbox("", list(size_map.keys()),
                                index=list(size_map.keys()).index(size_label),
                                label_visibility="collapsed", key="ps")

            st.markdown("<hr style='margin:28px 0; border-color:#EEEEEE'>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:11px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#7A7469; margin-bottom:8px'>주요 관계</p>", unsafe_allow_html=True)
            rel_label = next((k for k, v in rel_map.items() if v == p["relationship"]), "친한 친구")
            sel_rel = st.radio("", list(rel_map.keys()),
                           index=list(rel_map.keys()).index(rel_label),
                           horizontal=True, label_visibility="collapsed", key="pr")

            st.markdown("<hr style='margin:28px 0; border-color:#EEEEEE'>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:11px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#7A7469; margin-bottom:8px'>공포 수용도</p>", unsafe_allow_html=True)
            hor_label = next((k for k, v in hor_map.items() if v == p["horrorTolerance"]), "일부 민감함 (공포 제외 권장)")
            sel_hor = st.radio("", list(hor_map.keys()),
                           index=list(hor_map.keys()).index(hor_label),
                           label_visibility="collapsed", key="ph")

            st.markdown("<hr style='margin:28px 0; border-color:#EEEEEE'>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:11px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#7A7469; margin-bottom:8px'>활동성 선호도</p>", unsafe_allow_html=True)
            act_label = next((k for k, v in act_map.items() if v == p["activityLevel"]), "보통")
            sel_act = st.radio("", list(act_map.keys()),
                           index=list(act_map.keys()).index(act_label),
                           label_visibility="collapsed", key="pa")

            st.markdown("<hr style='margin:28px 0; border-color:#EEEEEE'>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:11px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#7A7469; margin-bottom:8px'>1인당 예산</p>", unsafe_allow_html=True)
            bud_label = next((k for k, v in bud_map.items() if v == p["budget"]), "~2만원")
            sel_bud = st.selectbox("", list(bud_map.keys()),
                               index=list(bud_map.keys()).index(bud_label),
                               label_visibility="collapsed", key="pb")



            if st.button("저장하기  →", type="primary", use_container_width=True):
                st.session_state.persona = {
                    "groupSize": size_map[sel_size],
                    "relationship": rel_map[sel_rel],
                    "horrorTolerance": hor_map[sel_hor],
                    "activityLevel": act_map[sel_act],
                    "budget": bud_map[sel_bud],
                }
                st.session_state.persona_saved = True

            if st.session_state.persona_saved:
                st.success("내 정보가 저장되었습니다. AI 추천에서 이 정보가 자동으로 반영됩니다.")


# ─── Router ───────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "home":
    page_home()
elif page == "ai":
    page_ai()
elif page == "explore":
    page_explore()
elif page == "persona":
    page_persona()