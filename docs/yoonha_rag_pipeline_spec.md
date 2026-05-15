# Nolit 4단계 RAG 파이프라인 입출력 스펙 문서
## 보드게임 · 머더미스터리 추천 시스템

---

## 0. 문서 목적

본 문서는 Nolit 4단계에서 구현한 보드게임 및 머더미스터리 추천용 RAG 파이프라인의 입출력 구조, 내부 모듈 역할, 데이터 의존성, 실행 방법, 평가 결과를 정의한다.

본 파이프라인은 사용자의 자연어 요청과 그룹 조건을 기반으로 다음 과정을 수행한다.

```text
사용자 입력
→ 조건 정규화
→ 정보 충분 여부 판단
→ 조건 부족 시 역질문 생성
→ 조건 충분 시 RAG 검색
→ 감정 태그 필터링
→ LLM 또는 룰 기반 추천 생성
→ 최종 결과 반환
```

최종 출력은 프론트엔드 또는 백엔드에서 바로 사용할 수 있도록 다음 3개 필드로 정규화된다.

```python
{
    "answer": "추천 요약 텍스트",
    "games": [...],
    "next_question": "추가 조건을 묻는 역질문"
}
```

---

## 1. 최종 산출물 구조

4단계 제출 기준 산출물은 다음과 같다.

```text
recommender/
├── graph.py                         # LangGraph 그래프 정의 wrapper
├── yoonha_graph.py                  # 실제 LangGraph 구현
└── rag/
    ├── query_transformer.py          # 제출용 wrapper: 조건 → 자연어 쿼리 변환
    ├── hybrid_retriever.py           # 제출용 wrapper: BM25 + FAISS 하이브리드 검색
    ├── tag_filter.py                 # 제출용 wrapper: 감정 태그 필터링
    ├── generator.py                  # 제출용 wrapper: LLM 추천 생성 / 역질문
    ├── yoonha_query_transformer.py   # 실제 query transformer 구현
    ├── yoonha_hybrid_retriever.py    # 실제 hybrid retriever router 구현
    ├── yoonha_boardgame_retriever.py # BGG + 보드라이프 검색 구현
    ├── yoonha_mm_retriever.py        # 머더미스터리 검색 구현
    ├── yoonha_tag_filter.py          # 실제 tag filter 구현
    ├── yoonha_generator.py           # 실제 generator 구현
    └── yoonha_emotion_tagger.py      # 감정 태그 전처리 스크립트
```

제출 요구사항의 파일명과 기존 구현 파일명을 모두 만족시키기 위해 wrapper 파일을 추가한다.

---

## 2. 전체 파이프라인 흐름

```text
graph.py / yoonha_graph.py
    ↓
query_transformer.py / yoonha_query_transformer.py
    ↓
hybrid_retriever.py / yoonha_hybrid_retriever.py
    ↓
tag_filter.py / yoonha_tag_filter.py
    ↓
generator.py / yoonha_generator.py
```

각 모듈의 역할은 다음과 같다.

| 단계 | 제출 파일 | 실제 구현 파일 | 역할 |
|---|---|---|---|
| LangGraph 그래프 | `recommender/graph.py` | `recommender/yoonha_graph.py` | LangGraph 노드·엣지 구성, 입력 adapter, 출력 정규화 |
| 쿼리 변환 | `recommender/rag/query_transformer.py` | `recommender/rag/yoonha_query_transformer.py` | 그룹 조건과 자연어 요청을 검색용 쿼리로 변환 |
| 하이브리드 검색 | `recommender/rag/hybrid_retriever.py` | `recommender/rag/yoonha_hybrid_retriever.py` | BM25 + FAISS 기반 RRF 검색 |
| 보드게임 검색 | - | `recommender/rag/yoonha_boardgame_retriever.py` | BGG + 보드라이프 데이터 검색 및 메타데이터 가중치 적용 |
| 머더미스터리 검색 | - | `recommender/rag/yoonha_mm_retriever.py` | 머더미스터리로그 데이터 검색 및 메타데이터 가중치 적용 |
| 감정 태그 필터링 | `recommender/rag/tag_filter.py` | `recommender/rag/yoonha_tag_filter.py` | 공포도, 난이도, 분위기, 관계 태그 기반 필터링 |
| 추천 생성 | `recommender/rag/generator.py` | `recommender/rag/yoonha_generator.py` | OpenAI API 또는 룰 기반 추천 문장 및 역질문 생성 |

---

## 3. 최종 인터페이스

최종 인터페이스는 두 가지 방식으로 사용할 수 있다.

1. `graph.invoke()`
2. `run_pipeline()`

4단계 제출 스펙 기준으로 가장 중요한 인터페이스는 `graph.invoke()`이다.

---

## 3-1. `graph.invoke()` 인터페이스

### Import

```python
from recommender.graph import graph
```

### 기본 입력 예시

```python
result = graph.invoke({
    "query": "4명이서 할 보드게임",
    "category": "boardgame"
})
```

### 확장 입력 예시

```python
result = graph.invoke({
    "query": "4명이서 할 전략 보드게임",
    "category": "boardgame",
    "group": {
        "headcount": 4,
        "play_time": 120,
        "weight_pref": "heavy",
        "category": "Strategy",
        "relation": "friend"
    },
    "use_api": False
})
```

### 입력 필드

| 필드 | 타입 | 필수 | 설명 | 예시 |
|---|---|---:|---|---|
| `query` | str | 필수 | 사용자 자연어 요청 | `"4명이서 할 보드게임"` |
| `category` | str | 필수 | 추천 도메인 | `"boardgame"` 또는 `"murdermystery"` |
| `group` | dict | 선택 | 이미 파싱된 그룹 조건 | `{"headcount": 4}` |
| `use_api` | bool | 선택 | OpenAI API 사용 여부. 기본값 `True` | `False` |

### 출력 예시

```python
{
    "answer": "4명으로 구성된 그룹에게 적합한 보드게임을 추천합니다.",
    "games": [
        {
            "title": "브라스: 버밍엄",
            "reason": "4명이 플레이하기에 적합한 인원 범위이며, 전략적 깊이가 있는 고난도 게임입니다.",
            "matched_tags": ["고난이도", "전략"],
            "final_score": 115.57,
            "emotion_tags": ["고난이도", "전략"],
            "source": "boardlife",
            "avg_rating": 8.5,
            "min_players": 2,
            "max_players": 4,
            "image": null
        }
    ],
    "next_question": "추가로 피하고 싶은 요소나 선호하는 분위기가 있나요?"
}
```

---

## 3-2. `run_pipeline()` 인터페이스

`run_pipeline()`은 기존 내부 개발 및 평가 코드와의 호환을 위한 함수이다.

### Import

```python
from recommender.graph import run_pipeline
```

### 입력 예시

```python
result = run_pipeline(
    user_text="4명이서 할 전략 보드게임 추천해줘",
    group={
        "headcount": 4,
        "play_time": 120,
        "weight_pref": "heavy",
        "category": "Strategy",
        "mechanism": "Market",
        "horror_tolerance": 2,
        "relation": "friend"
    },
    category="boardgame",
    use_api=False
)
```

### 입력 필드

| 필드 | 타입 | 필수 | 설명 | 예시 |
|---|---|---:|---|---|
| `user_text` | str | 필수 | 사용자 자연어 입력 | `"4명이서 할 전략 보드게임 추천해줘"` |
| `group` | dict | 선택 | 그룹 조건 | `{"headcount": 4}` |
| `category` | str | 필수 | 추천 도메인 | `"boardgame"` |
| `use_api` | bool | 선택 | OpenAI API 사용 여부. 기본값 `True` | `False` |

---

## 4. 최종 출력 스펙

최종 출력은 항상 아래 구조로 반환된다.

```python
{
    "answer": str,
    "games": list[dict],
    "next_question": str
}
```

### 출력 필드 상세

| 필드 | 타입 | 설명 |
|---|---|---|
| `answer` | str | 사용자에게 보여줄 추천 요약 텍스트 |
| `games` | list[dict] | 추천된 게임 또는 작품 목록 |
| `next_question` | str | 조건 보완 또는 추가 선호 확인을 위한 후속 질문 |

### `games` 내부 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `title` | str | 게임 또는 작품 제목 |
| `reason` | str | 추천 이유 |
| `matched_tags` | list[str] | 사용자 조건과 매칭된 감정 태그 |
| `final_score` | float | 필터링 및 점수 조정 후 최종 점수 |
| `emotion_tags` | list[str] | 아이템에 부여된 감정 태그 |
| `source` | str | 데이터 출처. 예: `bgg`, `boardlife`, `murdermysterylog` |
| `avg_rating` | float 또는 None | 평균 평점 |
| `min_players` | int 또는 float 또는 None | 최소 플레이 인원 |
| `max_players` | int 또는 float 또는 None | 최대 플레이 인원 |
| `image` | str 또는 None | 이미지 URL |

---

## 5. 그룹 조건 스펙

`group`은 사용자의 그룹 상황을 구조화한 dict이다.

### 공통 필드

| 필드 | 타입 | 설명 | 예시 |
|---|---|---|---|
| `headcount` | int | 인원 수 | `4` |
| `play_time` | int | 최대 플레이 시간. 단위는 분 | `120` |
| `horror_tolerance` | int | 공포 수용도. `0=불가`, `1=약간 가능`, `2=가능` | `0` |
| `relation` | str | 그룹 관계 | `"friend"` |

### `relation` 값

| 값 | 의미 |
|---|---|
| `first_meeting` | 처음 만나는 사이 |
| `couple` | 커플 또는 데이트 |
| `friend` | 친구 |
| `coworker` | 직장동료 또는 팀 |

### 보드게임 전용 필드

| 필드 | 타입 | 설명 | 예시 |
|---|---|---|---|
| `weight_pref` | str | 선호 난이도. `light`, `medium`, `heavy` | `"heavy"` |
| `category` | str | BGG/보드라이프 카테고리 | `"Strategy"`, `"Party"`, `"Economic"` |
| `mechanism` | str | 보드게임 메커니즘 | `"Worker Placement"`, `"Deck Building"`, `"Market"` |
| `source_pref` | str | 데이터 출처 선호. 한국 유저 반응 중심이면 `korean` | `"korean"` |

### 머더미스터리 전용 필드

| 필드 | 타입 | 설명 | 예시 |
|---|---|---|---|
| `difficulty_pref` | str | 선호 난이도. `light`, `medium`, `heavy` | `"light"` |
| `scene_category` | str | 작품 유형 조건. 데이터에 있을 때 적용 | `"보드게임형"` |
| `area` | str | 지역 필터. 향후 확장용 | `"서울"` |
| `location` | str | 세부 지역 필터. 향후 확장용 | `"강남"` |

---

## 6. 입력 자동 파싱 규칙

`graph.invoke()`는 `query`에서 일부 조건을 자동 추출한다.

### 예시

```python
graph.invoke({
    "query": "4명이서 할 보드게임",
    "category": "boardgame"
})
```

내부적으로 다음과 같이 변환된다.

```python
{
    "user_text": "4명이서 할 보드게임",
    "category": "boardgame",
    "group": {
        "headcount": 4
    }
}
```

### 자동 추출 가능한 항목

| 항목 | 예시 입력 | 추출 결과 |
|---|---|---|
| 인원 수 | `"4명이서"`, `"4인"` | `headcount=4` |
| 플레이 시간 | `"2시간 안에"`, `"90분 이하"` | `play_time=120`, `play_time=90` |
| 난이도 | `"쉬운"`, `"입문용"`, `"헤비한"` | `weight_pref="light"` 또는 `"heavy"` |
| 공포 수용도 | `"공포 싫어"`, `"안 무서운"` | `horror_tolerance=0` |
| 관계 | `"친구랑"`, `"데이트"`, `"직장동료"` | `relation="friend"` 등 |
| 보드게임 장르 | `"전략"`, `"파티"`, `"협력"` | `category="Strategy"` 등 |

---

## 7. 정보 충분 여부 판단 및 역질문

파이프라인은 추천을 실행하기 전에 최소 조건이 충분한지 확인한다.

### 최소 조건

| 조건 | 설명 |
|---|---|
| `query` 존재 | 사용자 요청이 있어야 함 |
| `category` 유효 | `"boardgame"` 또는 `"murdermystery"` |
| `headcount` 존재 | 그룹 인원 수가 있어야 함 |

### 조건이 부족한 경우

RAG 검색을 실행하지 않고 바로 역질문을 반환한다.

입력:

```python
graph.invoke({
    "query": "보드게임 추천해줘",
    "category": "boardgame"
})
```

출력:

```python
{
    "answer": "추천을 정확히 하기 위해 조건이 조금 더 필요합니다.",
    "games": [],
    "next_question": "몇 명이서 함께할 예정인가요?"
}
```

---

## 8. 내부 모듈 입출력

---

## 8-1. `query_transformer.transform()`

그룹 조건과 자연어 입력을 BM25 및 FAISS 검색에 사용할 수 있는 쿼리 형태로 변환한다.

### Import

```python
from recommender.rag.query_transformer import transform
```

### 입력

```python
result = transform(
    user_text="4명이서 할 전략 보드게임",
    group={
        "headcount": 4,
        "play_time": 120,
        "weight_pref": "heavy",
        "category": "Strategy"
    },
    category="boardgame"
)
```

### 출력

```python
{
    "query_text": "4명이서 할 전략 보드게임 4인 무거운 복잡 전략 고급 어려운 전략 strategy",
    "query_filter": {
        "players": 4,
        "playing_time": 120,
        "weight_pref": "heavy",
        "category": "Strategy",
        "source_pref": "korean"
    },
    "emotion_tags": ["웃음", "친목용"],
    "anchor_titles": ["Brass: Birmingham", "Twilight Struggle", "Terra Mystica"]
}
```

### 출력 필드

| 필드 | 소비자 | 설명 |
|---|---|---|
| `query_text` | `hybrid_retriever` | BM25 검색용 자연어 쿼리 |
| `query_filter` | `hybrid_retriever` | hard filter 및 metadata reranking 조건 |
| `emotion_tags` | `tag_filter` | 감정 태그 필터링에 사용 |
| `anchor_titles` | `hybrid_retriever` | dense embedding 생성에 사용 |

---

## 8-2. `hybrid_retriever.retrieve()`

BM25와 FAISS dense retrieval 결과를 RRF 방식으로 결합한다.

### Import

```python
from recommender.rag.hybrid_retriever import retrieve, get_embedding
```

### 입력

```python
query_vector = get_embedding(
    ["Brass: Birmingham"],
    category="boardgame"
)

items = retrieve(
    query_text="전략 strategy heavy",
    query_filter={
        "players": 4,
        "playing_time": 120
    },
    query_vector=query_vector,
    category="boardgame",
    topk=50
)
```

### 출력

```python
[
    {
        "title": "브라스: 버밍엄",
        "source": "boardlife",
        "avg_rating": 8.5,
        "rrf_score": 0.032787,
        "meta_score": 45.0,
        "total_score": 115.57,
        "bm25_rank": 5,
        "dense_rank": 1,
        "emotion_tags": ["분위기좋음", "고난이도"]
    }
]
```

### 지원 검색 방식

| 함수 | 설명 |
|---|---|
| `retrieve()` | RRF 기반 BM25 + Dense 하이브리드 검색 |
| `retrieve_bm25()` | BM25 단독 검색 |
| `retrieve_dense()` | FAISS Dense 단독 검색 |
| `retrieve_vanilla()` | 필터 통과 후 평점 또는 메타 점수 기반 정렬 |

---

## 8-3. `tag_filter.filter_and_score()`

감정 태그 기반으로 검색 결과를 필터링하고 점수를 보정한다.

### Import

```python
from recommender.rag.tag_filter import filter_and_score
```

### 입력

```python
filtered = filter_and_score(
    items=items,
    emotion_tags=[
        "입문용",
        "분위기좋음",
        "처음만나는사이추천"
    ],
    horror_tolerance=0,
    emotion_weight=5.0
)
```

### 동작 방식

| 단계 | 설명 |
|---|---|
| 공포 하드 필터 | `horror_tolerance=0`이면 공포 관련 태그가 강한 아이템 제외 |
| 긍정 태그 매칭 | 요청 태그와 아이템 태그의 교집합에 보너스 부여 |
| 부정 태그 패널티 | 고난이도, 복잡함 등 조건과 맞지 않는 태그 감점 |
| 최종 점수 계산 | `total_score + tag_score` 기반 `final_score` 산출 |

### 출력에 추가되는 필드

| 필드 | 설명 |
|---|---|
| `emotion_match_score` | 긍정 태그 매칭 점수 |
| `negative_score` | 부정 태그 패널티 |
| `final_score` | 최종 추천 점수 |

---

## 8-4. `generator.generate()`

검색 결과를 기반으로 LLM 추천 문장과 후속 질문을 생성한다.

### Import

```python
from recommender.rag.generator import generate, generate_without_api
```

### OpenAI API 사용

```python
result = generate(
    items=filtered,
    group={
        "headcount": 4,
        "weight_pref": "heavy"
    },
    category="boardgame",
    emotion_tags=["친목용"]
)
```

### API 없이 룰 기반 생성

```python
result = generate_without_api(
    items=filtered,
    group={
        "headcount": 4,
        "weight_pref": "heavy"
    },
    category="boardgame",
    emotion_tags=["친목용"]
)
```

### 참고

`graph.invoke()`와 `run_pipeline()`의 최종 출력은 `graph.py` / `yoonha_graph.py`에서 다음 구조로 정규화된다.

```python
{
    "answer": "...",
    "games": [...],
    "next_question": "..."
}
```

---

## 9. 하드 필터 및 메타데이터 가중치 정책

검색 품질 개선을 위해 하드 필터와 메타데이터 기반 가중치를 적용한다.

### 하드 필터

하드 필터는 조건을 명확히 벗어나는 결과를 제외한다.

| 도메인 | 필드 | 설명 |
|---|---|---|
| 보드게임 | `min_players`, `max_players` | 쿼리 인원이 플레이 가능 범위 밖이면 제외 |
| 보드게임 | `playing_time`, `max_time` | 쿼리 최대 시간을 초과하면 제외 |
| 머더미스터리 | `min_players`, `max_players` | 쿼리 인원이 플레이 가능 범위 밖이면 제외 |
| 머더미스터리 | `max_time`, `play_time` | 쿼리 최대 시간을 초과하면 제외 |
| 머더미스터리 | `scene_category` | 유형 조건이 명시된 경우 불일치 결과 제외 |

### 메타데이터 가중치

하드 필터를 통과한 결과에 대해 다음 항목을 기반으로 reranking을 수행한다.

| 도메인 | 필드 | 가중치 방향 |
|---|---|---|
| 보드게임 | `category` | 쿼리 카테고리와 매칭되면 가산 |
| 보드게임 | `mechanism` | 쿼리 메커니즘과 매칭되면 가산 |
| 보드게임 | `avg_rating` | 소스별 정규화 후 높을수록 가산 |
| 보드게임 | `weight` | `light`는 낮은 weight, `heavy`는 높은 weight 우선 |
| 보드게임 | `recommended_players`, `best_players` | 쿼리 인원과 일치하면 가산 |
| 보드게임 | `category_rank`, `rank` | 낮은 순위일수록 가산 |
| 보드게임 | `source` | 한국어/한국 유저 기반 쿼리는 boardlife 우선 |
| 머더미스터리 | `difficulty` | `light`는 낮은 난이도, `heavy`는 높은 난이도 우선 |
| 머더미스터리 | `horror` | `low`는 낮은 공포도, `high`는 높은 공포도 우선 |
| 머더미스터리 | `rating` | 높을수록 가산 |

### 주의사항

| 항목 | 설명 |
|---|---|
| 평점 스케일 | BGG는 1~10, 보드라이프는 1~5로 다르므로 소스별 정규화 후 사용 |
| None 처리 | `None` 또는 `NaN`은 0점이 아니라 데이터 없음으로 처리 |
| weight | 0~5 범위이며 높을수록 복잡하고 어려운 보드게임 |
| horror | 0~5 범위이며 높을수록 공포 요소가 강함 |
| difficulty | 머더미스터리 난이도는 데이터 소스별 스케일 차이를 고려해 사용 |

---

## 10. 감정 태그 시스템

감정 태그는 단순 평점 기반 추천의 한계를 보완하기 위한 라이트 태깅 시스템이다.

### 태깅 차원

| 차원 | 태그 예시 | 사용 목적 |
|---|---|---|
| 공포 | `공포있음`, `공포높음`, `공포없음` | 공포 민감 인원 보호 |
| 난이도 | `고난이도`, `복잡함`, `입문용`, `가볍게즐길수있음` | 초보자/고급자 그룹 분리 |
| 분위기 | `분위기좋음`, `분위기별로`, `웃음`, `대화유도` | 어색함 완화, 친목 목적 반영 |
| 관계 | `처음만나는사이추천`, `데이트추천`, `친목용` | 그룹 관계 맥락 반영 |
| 인원 | `인원적절` | 실제 플레이 인원 적합성 보정 |

### 태깅 방식

| 데이터 | 방식 | 설명 |
|---|---|---|
| 머더미스터리로그 | 리뷰 텍스트 키워드 매칭 | 리뷰에서 공포, 난이도, 분위기 키워드 추출 |
| 보드게임 BGG/보드라이프 | 메타데이터 기반 | weight, avg_rating, player range 기반 태깅 |

### 향후 확장

방탈출 데이터는 향후 확장 대상으로 분리하며, 본 4단계 RAG 파이프라인 검증 범위에는 포함하지 않는다.

---

## 11. 데이터 의존성

본 프로젝트의 대용량 데이터 파일은 Git에 포함하지 않고 외부 공유 폴더를 통해 관리한다.

### 4단계 테스트 기준 필수 데이터 파일

```text
data/
├── faiss_bgg_stats.index
├── faiss_bgg_stats_meta.json
├── faiss_bgg_reviews_meta.json
├── faiss_boardlife_stats.index
├── faiss_boardlife_stats_meta.json
├── faiss_boardlife_reviews_meta.json
├── faiss_murdermysterylog.index
└── faiss_murdermysterylog_meta.json
```

### 실제 테스트 기준 로딩 확인

E2E 테스트에서 다음 데이터 로딩이 확인되었다.

```text
BGG: 9,913개
보드라이프: 3,322개
머더미스터리: 281개
```

### 향후 확장 예정 데이터

다음 데이터는 향후 확장 대상으로 관리한다.

```text
data/
├── faiss_murmynow.index
├── faiss_murmynow_meta.json
├── faiss_bbabang_stats.index
├── faiss_bbabang_stats_metadata.json
├── faiss_bbabang_reviews.index
└── faiss_bbabang_reviews_metadata.json
```

---

## 12. 환경 설정

### 패키지 설치

```bash
uv pip install faiss-cpu rank-bm25 numpy langgraph python-dotenv openai
```

또는 pip 사용 시:

```bash
pip install faiss-cpu rank-bm25 numpy langgraph python-dotenv openai
```

### `.env`

```env
OPENAI_API_KEY=sk-...
```

`use_api=True`가 기본값이므로 OpenAI API 키가 필요하다.

다만 평가 및 로컬 검증 시에는 다음과 같이 `use_api=False`를 사용할 수 있다.

```python
graph.invoke({
    "query": "4명이서 할 보드게임",
    "category": "boardgame",
    "use_api": False
})
```

---

## 13. 실행 및 테스트

### graph.invoke E2E 테스트

프로젝트 루트에서 실행한다.

```bash
python -m recommender.eval.yoonha_test_graph
```

테스트 항목은 다음과 같다.

| 테스트 | 검증 내용 |
|---|---|
| `graph.invoke` 테스트 | 제출용 입력 스펙 동작 여부 |
| `run_pipeline` 테스트 | 기존 내부 호출 방식 호환 여부 |
| Clarifying Question 테스트 | 조건 부족 시 역질문 생성 여부 |
| 머더미스터리 테스트 | `murdermystery` 도메인 검색 및 생성 여부 |

### 통합 평가 테스트

프로젝트 루트에서 실행한다.

```bash
python -m compileall recommender
python -m recommender.eval.yoonha_eval_pipeline
```

평가 항목은 다음과 같다.

| 항목 | 설명 |
|---|---|
| query_transformer 테스트 | 자연어 입력이 검색 쿼리로 적절히 변환되는지 확인 |
| retriever 비교 | RRF, BM25, Dense, Vanilla 방식 비교 |
| hard filter 검증 | 인원/시간 조건 위반 결과가 제거되는지 확인 |
| preference direction 검증 | 난이도/공포도 선호 방향성이 반영되는지 확인 |
| source weighting 검증 | 한국어 쿼리에서 boardlife 결과가 반영되는지 확인 |
| tag_filter 테스트 | 감정 태그 필터링 전후 결과 비교 |
| generator 테스트 | 추천 문장과 역질문 생성 확인 |
| graph E2E 테스트 | 전체 파이프라인 연결 확인 |
| Precision@K 비교 | 검색 방식별 정량 평가 |

### 실제 테스트 결과 요약

```text
compileall 성공
query_transformer 테스트 통과
retriever 비교 테스트 실행 완료
hard filter 검증 PASS
preference direction 검증 PASS
source weighting 검증 PASS
tag_filter 테스트 완료
generator 테스트 완료
boardgame graph E2E 추천 5개 생성
murdermystery graph E2E 추천 5개 생성
전체 평가 완료
```

---

## 14. 평가 결과 요약

`yoonha_eval_pipeline.py`는 RAG 검색 품질을 평가하기 위한 별도 평가 스크립트이다.

### Precision@10 결과

| 방식 | 평균 Precision@10 |
|---|---:|
| RRF, BM25 + Dense | 0.100 |
| Dense 단독 | 0.120 |
| BM25 단독 | 0.040 |
| Vanilla | 0.040 |

Dense 단독의 평균 Precision@10이 일부 테스트 케이스에서 높게 나타났지만, 실제 운영 기본 방식은 BM25와 Dense Retrieval을 결합하고 하드 필터 및 메타데이터 가중치를 반영하는 RRF 방식을 사용한다.

---

## 15. 예외 및 fallback 정책

| 상황 | 처리 방식 |
|---|---|
| `query` 없음 | RAG 검색 없이 역질문 반환 |
| `category` 불명확 | 보드게임/머더미스터리 중 선택 요청 |
| `headcount` 없음 | 인원 수 역질문 반환 |
| FAISS 또는 데이터 파일 없음 | 빈 검색 결과 반환 후 fallback 메시지 생성 |
| OpenAI API 실패 | 룰 기반 generator로 대체 |
| 검색 결과 없음 | 추천 불가 메시지와 추가 조건 질문 반환 |
| `None` 또는 `NaN` 메타데이터 | 0점으로 간주하지 않고 데이터 없음으로 처리 |

---

## 16. 최종 사용 예시

### 보드게임 추천

```python
from recommender.graph import graph

result = graph.invoke({
    "query": "4명이서 할 전략 보드게임",
    "category": "boardgame",
    "use_api": False
})

print(result["answer"])
print(result["games"])
print(result["next_question"])
```

### 머더미스터리 추천

```python
from recommender.graph import graph

result = graph.invoke({
    "query": "6명이서 할 쉬운 머더미스터리",
    "category": "murdermystery",
    "use_api": False
})

print(result["answer"])
print(result["games"])
print(result["next_question"])
```

### 조건 부족 시 역질문

```python
from recommender.graph import graph

result = graph.invoke({
    "query": "보드게임 추천해줘",
    "category": "boardgame"
})

print(result)
```

출력:

```python
{
    "answer": "추천을 정확히 하기 위해 조건이 조금 더 필요합니다.",
    "games": [],
    "next_question": "몇 명이서 함께할 예정인가요?"
}
```

---

## 17. 최종 산출물 체크리스트

| 산출물 | 제출 파일 | 구현 연결 | 상태 |
|---|---|---|---|
| 조건 → 쿼리 변환 | `recommender/rag/query_transformer.py` | `yoonha_query_transformer.py` wrapper | 완료 |
| BM25 + FAISS 검색 | `recommender/rag/hybrid_retriever.py` | `yoonha_hybrid_retriever.py` wrapper | 완료 |
| 감정 태그 필터링 | `recommender/rag/tag_filter.py` | `yoonha_tag_filter.py` wrapper | 완료 |
| LLM 추천 생성 / 역질문 | `recommender/rag/generator.py` | `yoonha_generator.py` wrapper | 완료 |
| LangGraph 그래프 | `recommender/graph.py` | `yoonha_graph.py` wrapper | 완료 |
| 입출력 스펙 문서 | `docs/yoonha_rag_pipeline_spec.md` | 본 문서 | 완료 |
| E2E 테스트 | `recommender/eval/yoonha_eval_pipeline.py` | graph.invoke/run_pipeline 검증 | 완료 |
| 데이터 파일 | 외부 공유 폴더 | Git 제외 | 완료 |
