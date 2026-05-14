"""
recommender/rag/loader.py

[ 역할 ]
  - 데이터소스별 CSV 로드 전담
  - config.yaml 설정을 읽어서 (texts, metas) 반환
  - bbabang은 stats + reviews 두 CSV를 합쳐서 처리 (fuzzy matching 포함)
  - embeddings.py에서 load_source(source_name) 으로 호출
"""

import json
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz
from typing import Any

from .config import get_config, get_data_dir


# ══════════════════════════════════════════════
# 1. 메인 진입점
# ══════════════════════════════════════════════

def load_source(source_name: str) -> tuple[list[str], list[dict]]:
    """
    config.yaml 기반 데이터소스 로드.
    embeddings.py에서 호출하는 메인 함수.

    Returns:
        (texts, metas)
    """
    cfg     = get_config()
    sources = cfg["sources"]

    if source_name not in sources:
        raise ValueError(
            f"알 수 없는 소스: '{source_name}'\n"
            f"사용 가능: {list(sources.keys())}"
        )

    src_cfg  = sources[source_name]
    data_dir = get_data_dir()
    engine   = src_cfg.get("engine", "openai")

    print(f"  [{source_name}] CSV 로드 중... (engine={engine})")

    # bbabang은 별도 처리
    if source_name == "bbabang_stats":
        return _load_bbabang_stats(src_cfg, data_dir)
    if source_name == "bbabang_reviews":
        return _load_bbabang_reviews(src_cfg, data_dir)

    # OpenAI 계열
    csv_path = data_dir / src_cfg["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 없음: {csv_path}")

    # text + metadata(JSON) 컬럼 구조
    if src_cfg.get("text_col") == "text" and "meta_cols" not in src_cfg:
        return _load_json_meta(csv_path)

    # text_col + meta_cols 구조
    return _load_with_meta_cols(csv_path, src_cfg)


# ══════════════════════════════════════════════
# 2. OpenAI 계열 로더
# ══════════════════════════════════════════════

def _load_json_meta(csv_path: Path) -> tuple[list, list]:
    """
    text / metadata(JSON 문자열) 컬럼 구조.
    bgg_stats, bgg_reviews, murmynow 사용.
    """
    df    = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    texts = df["text"].fillna(" ").astype(str).tolist()
    metas = df["metadata"].apply(_safe_json).tolist()
    return texts, metas


def _load_with_meta_cols(csv_path: Path, src_cfg: dict) -> tuple[list, list]:
    """
    text_col + meta_cols 컬럼 구조.
    boardlife_stats, boardlife_reviews, murdermysterylog 사용.
    """
    df       = pd.read_csv(csv_path)
    text_col = src_cfg["text_col"]
    texts    = df[text_col].fillna(" ").astype(str).tolist()

    if "meta_cols" in src_cfg:
        meta_cols = [c for c in src_cfg["meta_cols"] if c in df.columns]
    else:
        meta_cols = [c for c in df.columns if c != text_col]

    metas = df[meta_cols].to_dict(orient="records")
    return texts, metas


# ══════════════════════════════════════════════
# 3. bbabang 전용 로더 (HuggingFace)
# ══════════════════════════════════════════════

def _load_bbabang_stats(src_cfg: dict, data_dir: Path) -> tuple[list, list]:
    """
    bbabang stats 로드.
    - stats + reviews CSV fuzzy matching으로 avg_headcount 병합
    - build_stats_document()로 임베딩용 텍스트 생성
    """
    df_stats   = pd.read_csv(data_dir / src_cfg["csv_stats"])
    df_reviews = pd.read_csv(data_dir / src_cfg["csv_reviews"], low_memory=False)
    print(f"    stats: {len(df_stats)}개 / reviews: {len(df_reviews)}개")

    # 리뷰 집계
    review_agg = _aggregate_reviews(df_reviews)

    # fuzzy matching으로 stats ↔ 리뷰 매핑
    df = _fuzzy_merge(df_stats, review_agg)

    # 임베딩 텍스트 생성
    texts = df.apply(_build_stats_document, axis=1).tolist()

    # 메타데이터 생성
    metas = []
    for i, row in df.reset_index(drop=True).iterrows():
        metas.append({
            "id"           : i,
            "source"       : "bbabang",
            "title"        : row["title"],
            "store_name"   : row["store_name"],
            "area"         : row.get("area"),
            "location"     : row.get("location"),
            "playing_time" : _safe_int(row.get("playing_time")),
            "max_players"  : _safe_int(row.get("max_players")),
            "price"        : _safe_int(row.get("price")),
            "difficulty"   : _safe_float(row.get("difficulty")),
            "horror"       : _safe_float(row.get("horror")),
            "activity"     : _safe_float(row.get("activity")),
            "satisfaction" : _safe_float(row.get("satisfaction")),
            "puzzle"       : _safe_float(row.get("puzzle")),
            "story"        : _safe_float(row.get("story")),
            "interior"     : _safe_float(row.get("interior")),
            "production"   : _safe_float(row.get("production")),
            "avg_headcount": _safe_float(row.get("avg_headcount")),
        })

    return texts, metas


def _load_bbabang_reviews(src_cfg: dict, data_dir: Path) -> tuple[list, list]:
    """
    bbabang reviews 로드.
    - 30자 이상 유효 리뷰 필터링
    - title + store 기준 10개씩 청크 생성
    """
    df_reviews = pd.read_csv(data_dir / src_cfg["csv_reviews"], low_memory=False)

    # 유효 리뷰 필터링 (30자 이상)
    df_valid = (
        df_reviews[df_reviews["review_text"].astype(str).str.len() >= 30]
        .copy()
        .reset_index(drop=True)
    )
    print(f"    유효 리뷰: {len(df_valid)}개")

    # 청크 생성 (title + store 기준 10개씩)
    chunks = []
    for (title, store), group in df_valid.groupby(["title", "store_name"]):
        reviews = group["review_text"].astype(str).tolist()
        for i in range(0, len(reviews), 10):
            chunk_texts = reviews[i:i + 10]
            chunk_doc   = f"테마명: {title} | 매장: {store} | 후기: {' '.join(chunk_texts)}"
            chunks.append({
                "title"       : title,
                "store_name"  : store,
                "source"      : "bbabang",
                "chunk_index" : i // 10,
                "document"    : chunk_doc,
            })

    print(f"    총 청크: {len(chunks)}개")

    texts = [c["document"] for c in chunks]
    metas = [
        {
            "id"          : i,
            "source"      : c["source"],
            "title"       : c["title"],
            "store_name"  : c["store_name"],
            "chunk_index" : c["chunk_index"],
            "document"    : c["document"],
        }
        for i, c in enumerate(chunks)
    ]
    return texts, metas


# ══════════════════════════════════════════════
# 4. bbabang 내부 유틸
# ══════════════════════════════════════════════

def _aggregate_reviews(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """title + store_name 기준 avg_headcount 집계"""
    def summarize(group):
        valid = group[group["review_headcount"] > 0]["review_headcount"]
        avg   = valid.mean() if len(valid) > 0 else None
        return pd.Series({"avg_headcount": round(avg, 2) if avg else None})

    return df_reviews.groupby(["title", "store_name"]).apply(summarize).reset_index()


def _fuzzy_merge(df_stats: pd.DataFrame, review_agg: pd.DataFrame) -> pd.DataFrame:
    """fuzzy matching으로 stats ↔ review_agg 병합"""
    review_keys = [
        f"{t}_{s}"
        for t, s in zip(review_agg["title"], review_agg["store_name"])
    ]

    def match(row):
        query  = f"{row['title']}_{row['store_name']}"
        result = process.extractOne(query, review_keys, scorer=fuzz.ratio)
        return result[0] if result and result[1] >= 85 else None

    df_stats["match_key"] = df_stats.apply(match, axis=1)

    unmatched = df_stats[df_stats["match_key"].isna()]
    if len(unmatched) > 0:
        print(f"    fuzzy 매칭 실패: {len(unmatched)}개")

    review_agg["match_key"] = [
        f"{t}_{s}"
        for t, s in zip(review_agg["title"], review_agg["store_name"])
    ]
    return df_stats.merge(
        review_agg.drop(columns=["title", "store_name"]),
        on="match_key",
        how="left"
    )


def _build_stats_document(row) -> str:
    """stats 임베딩용 텍스트 생성"""
    parts = [f"테마명: {row['title']}"]

    if pd.notna(row.get("description")):
        desc = str(row["description"]).replace(row["title"], "").strip()
        if desc:
            parts.append(f"설명: {desc}")

    scores = []
    for col, label in [
        ("difficulty", "난이도"), ("horror", "공포도"), ("activity", "활동성"),
        ("satisfaction", "만족도"), ("puzzle", "퍼즐"), ("story", "스토리"),
    ]:
        if pd.notna(row.get(col)):
            scores.append(f"{label} {row[col]}")
    if scores:
        parts.append(", ".join(scores))

    if pd.notna(row.get("playing_time")):
        parts.append(f"플레이타임 {int(row['playing_time'])}분")
    if pd.notna(row.get("max_players")):
        parts.append(f"최대 인원 {int(row['max_players'])}명")
    if pd.notna(row.get("area")):
        parts.append(f"지역 {row['area']} {row['location']}")
    if pd.notna(row.get("avg_headcount")) and row["avg_headcount"] > 0:
        parts.append(f"평균 플레이 인원 {round(row['avg_headcount'], 1)}명")

    return " | ".join(parts)


# ══════════════════════════════════════════════
# 5. 공통 유틸
# ══════════════════════════════════════════════

def _safe_json(value: Any) -> dict:
    try:
        return json.loads(value) if isinstance(value, str) else (value or {})
    except Exception:
        return {}

def _safe_int(value) -> int | None:
    try:
        return None if pd.isna(value) else int(value)
    except Exception:
        return None

def _safe_float(value) -> float | None:
    try:
        return None if pd.isna(value) else float(value)
    except Exception:
        return None
