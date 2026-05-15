"""
yoonha_hybrid_retriever.py
category 기반 라우터 — yoonha_graph.py 에서 이 파일만 호출하면 됨.

====================================================================
[역할]
    그래프(yoonha_graph.py)와 개별 retriever 사이의 중간 계층.
    category 값("boardgame" | "murdermystery")에 따라
    적절한 retriever 모듈로 요청을 라우팅한다.

[설계 의도]
    - 그래프 코드가 개별 retriever를 직접 알 필요 없이
      이 파일의 retrieve() 하나만 호출하면 된다.
    - 새로운 카테고리(예: 방탈출) 추가 시 이 파일에만
      라우팅 분기를 추가하면 그래프 코드는 수정 불필요.

[내부 라우팅]
    "boardgame"     → boardgame_retriever.retrieve()
    "murdermystery" → mm_retriever.retrieve()

[현재 미지원]
    방탈출(escape room) 카테고리 — 기획서에는 방탈출 페이지가 있으나
    retriever가 아직 구현되지 않았으므로 라우팅 분기 없음.

[제공 함수 — 4종 검색 + 1종 임베딩]
    retrieve()         : BM25 + FAISS RRF 하이브리드 검색 (주력)
    retrieve_bm25()    : BM25 단독 (디버깅/평가용)
    retrieve_dense()   : FAISS 단독 (디버깅/평가용)
    retrieve_vanilla() : 필터+평점순 (FAISS 장애 시 fallback)
    get_embedding()    : 앵커 타이틀 → 임베딩 벡터 변환
====================================================================
"""

import numpy as np
from pathlib import Path
import sys

# ─────────────────────────────────────────────
# 임포트 경로 보정
# ─────────────────────────────────────────────
# 직접 실행(python hybrid_retriever.py)과 패키지 임포트 둘 다 지원하기 위해
# 현재 파일이 위치한 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 보드게임 retriever 함수들을 _bg_ 접두사로 가져오기
from yoonha_boardgame_retriever import (
    retrieve as _bg_retrieve,
    retrieve_bm25 as _bg_bm25,
    retrieve_dense as _bg_dense,
    retrieve_vanilla as _bg_vanilla,
    get_embedding as _bg_embedding,
)

# 머더미스터리 retriever 함수들을 _mm_ 접두사로 가져오기
from yoonha_mm_retriever import (
    retrieve as _mm_retrieve,
    retrieve_bm25 as _mm_bm25,
    retrieve_dense as _mm_dense,
    retrieve_vanilla as _mm_vanilla,
    get_embedding as _mm_embedding,
)


# ─────────────────────────────────────────────
# 메인 라우팅 함수
# ─────────────────────────────────────────────
def retrieve(
    query_text: str,
    query_filter: dict,
    query_vector: np.ndarray,
    category: str,
    topk: int = 50,
) -> list[dict]:
    """
    카테고리에 맞는 retriever로 라우팅.

    [호출 흐름]
        graph.py → node_retrieve() → 여기 → boardgame_retriever / mm_retriever

    Args:
        query_text:    BM25용 자연어 쿼리 (query_transformer가 생성)
        query_filter:  hard_filter 조건 dict
                       보드게임: players, playing_time, weight_max, weight_pref,
                                category, mechanism
                       머더미스터리: players, max_time, area
        query_vector:  FAISS dense 검색용 벡터 (1, dim)
        category:      "boardgame" | "murdermystery"
        topk:          반환할 최대 아이템 수

    Returns:
        아이템 리스트 (total_score 내림차순)

    Raises:
        ValueError: category가 지원되지 않는 값일 때
    """
    if category == "boardgame":
        return _bg_retrieve(query_text, query_filter, query_vector, topk)
    elif category == "murdermystery":
        return _mm_retrieve(query_text, query_filter, query_vector, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}. 'boardgame' 또는 'murdermystery' 사용.")


# ─────────────────────────────────────────────
# BM25 단독 검색 라우터 (평가/디버깅용)
# ─────────────────────────────────────────────
def retrieve_bm25(
    query_text: str,
    query_filter: dict,
    category: str,
    topk: int = 50,
) -> list[dict]:
    """BM25 단독 검색 라우터."""
    if category == "boardgame":
        return _bg_bm25(query_text, query_filter, topk)
    elif category == "murdermystery":
        return _mm_bm25(query_text, query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}.")


# ─────────────────────────────────────────────
# Dense 단독 검색 라우터 (평가/디버깅용)
# ─────────────────────────────────────────────
def retrieve_dense(
    query_vector: np.ndarray,
    query_filter: dict,
    category: str,
    topk: int = 50,
) -> list[dict]:
    """Dense 단독 검색 라우터."""
    if category == "boardgame":
        return _bg_dense(query_vector, query_filter, topk)
    elif category == "murdermystery":
        return _mm_dense(query_vector, query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}.")


# ─────────────────────────────────────────────
# Vanilla 검색 라우터 (FAISS 장애 시 fallback)
# ─────────────────────────────────────────────
def retrieve_vanilla(
    query_filter: dict,
    category: str,
    topk: int = 50,
) -> list[dict]:
    """Vanilla 검색 라우터 — 필터 통과 후 평점 내림차순."""
    if category == "boardgame":
        return _bg_vanilla(query_filter, topk)
    elif category == "murdermystery":
        return _mm_vanilla(query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}.")


# ─────────────────────────────────────────────
# 앵커 임베딩 변환 라우터
# ─────────────────────────────────────────────
def get_embedding(titles: list[str], category: str) -> np.ndarray:
    """
    앵커 타이틀 리스트 → 평균 임베딩 벡터 반환 (shape: (1, dim)).

    [용도]
        query_transformer가 추출한 anchor_titles를 FAISS 검색용
        벡터로 변환. 카테고리별 인덱스 차원이 다를 수 있으므로
        적절한 retriever에 위임한다.

    Args:
        titles:   임베딩 기준 타이틀 리스트 (없으면 빈 리스트 [] 전달)
        category: "boardgame" | "murdermystery"
    """
    if category == "boardgame":
        return _bg_embedding(titles)
    elif category == "murdermystery":
        return _mm_embedding(titles)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}.")