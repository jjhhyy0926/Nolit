"""
OpenAI 임베딩 + FAISS 저장

[ 임베딩 대상 ]
  - murdermysterylog : reviews → faiss_murdermysterylog.index

[ 저장 결과 ]
  - .index (벡터) + _meta.json (메타데이터) 쌍으로 저장
  - 인덱스 번호로 벡터 ↔ 메타데이터 매핑
"""

import os
import json
import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv


# ────────────────────────────────────────────
# 0. 설정
# ────────────────────────────────────────────
load_dotenv()  # .env 파일에서 OPENAI_API_KEY 로드

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536차원
BATCH_SIZE      = 100                        # OpenAI API 한 번에 보낼 텍스트 수


# ────────────────────────────────────────────
# 1. 데이터 로드 및 준비
# ────────────────────────────────────────────

def load_murdermysterylog(path="data/murdermysterylog_final.csv"):
    """
    머더미스터리로그 데이터 로드.
    임베딩 텍스트 : reviews
    메타데이터    : url, name, rating, play_time, description,
                   시리즈, 제작, reviews, min_players, max_players, source
    """
    df = pd.read_csv(path)

    # 임베딩 텍스트: reviews만 사용
    texts = df["reviews"].fillna(" ").tolist()

    # 메타데이터 컬럼 목록
    meta_cols = [
        "url", "name", "rating", "play_time", "description",
        "시리즈", "제작", "reviews", "min_players", "max_players", "source"
    ]
    # 실제 CSV에 없는 컬럼은 자동으로 건너뜀
    meta_cols = [c for c in meta_cols if c in df.columns]

    meta = df[meta_cols].to_dict(orient="records")

    return texts, meta


# ────────────────────────────────────────────
# 2. OpenAI 임베딩 생성
# ────────────────────────────────────────────

def get_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    텍스트 리스트를 BATCH_SIZE 단위로 나눠서 OpenAI API 호출.
    빈 텍스트는 공백 1칸으로 대체 (API 에러 방지).
    반환값: 각 텍스트에 대한 임베딩 벡터 리스트 (1536차원)
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="임베딩 생성 중"):
        batch = texts[i : i + BATCH_SIZE]

        # 빈 텍스트 공백으로 대체 (OpenAI API는 빈 문자열 허용 안 함)
        batch = [t if t.strip() else " " for t in batch]

        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# ────────────────────────────────────────────
# 3. FAISS 인덱스 생성 및 저장
# ────────────────────────────────────────────

def save_index(
    embeddings: list[list[float]],
    metadata: list[dict],
    index_path: str,
    meta_path: str
):
    """
    임베딩 벡터로 FAISS 인덱스를 만들고 파일로 저장.
    - index_path : 벡터 저장 (.index)
    - meta_path  : 메타데이터 저장 (.json)
    인덱스 번호가 같으면 벡터 ↔ 메타데이터가 1:1 매핑됨.
    """
    # float32로 변환 (FAISS 요구 형식)
    vectors   = np.array(embeddings, dtype="float32")
    dimension = vectors.shape[1]  # text-embedding-3-small → 1536차원

    # FAISS 인덱스 생성 및 벡터 추가
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # 벡터 저장 (.index)
    faiss.write_index(index, index_path)

    # 메타데이터 저장 (.json)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"   ✅ {index_path}  ({index.ntotal}개 벡터)")
    print(f"   ✅ {meta_path}  ({len(metadata)}개 메타데이터)")


# ────────────────────────────────────────────
# 4. 메인 실행
# ────────────────────────────────────────────

def main():

    # ── 머더미스터리로그 ────────────────────────
    print("\n📂 murdermysterylog_final.csv 로드 중...")
    texts, meta = load_murdermysterylog()
    print(f"   → {len(texts)}개 로드 완료")

    print("🔄 임베딩 생성 중...")
    embeddings = get_embeddings(texts)

    print("💾 저장 중...")
    save_index(
        embeddings, meta,
        index_path="faiss_murdermysterylog.index",
        meta_path="faiss_murdermysterylog_meta.json"
    )

    print("\n🎉 임베딩 완료!")
    print("   faiss_murdermysterylog.index / faiss_murdermysterylog_meta.json")


if __name__ == "__main__":
    main()
