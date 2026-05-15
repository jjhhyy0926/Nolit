import pandas as pd
import numpy as np

df = pd.read_csv('bbabang_all_reviews.csv', low_memory=False)

# ── 1. 필요한 컬럼만 선택
keep_cols = [
    'title', 'store_name', 'review_text', 'review_headcount'
]
df = df[keep_cols]
print('[1] 필요 컬럼만 선택 완료')

# ── 2. review_headcount 0 → NaN
df.loc[df['review_headcount'] == 0, 'review_headcount'] = np.nan
print('[2] review_headcount 0 → NaN 처리 완료')

# ── 3. review_text #숫자 포함된 텍스트 전체 → NaN
hash_mask = df['review_text'].str.contains(r'#\d+', na=False)
df.loc[hash_mask, 'review_text'] = np.nan
print(f'[3] #숫자 포함 리뷰 → NaN 교체: {hash_mask.sum()}행')

# ── 4. review_text 클렌징
df['review_text'] = (
    df['review_text']
    .str.replace(r'\r\n|\r|\n', ' ', regex=True)
    .str.strip()
)

# ── 5. 5자 미만 리뷰 → NaN 교체
short_mask = df['review_text'].str.len() < 5
df.loc[short_mask, 'review_text'] = np.nan
print(f'[5] 5자 미만 리뷰 → NaN 교체: {short_mask.sum()}행')

# ── 6. 한글 없는 리뷰 → NaN 교체
korean_mask = ~df['review_text'].str.contains(r'[가-힣]', na=False)
df.loc[korean_mask, 'review_text'] = np.nan
print(f'[6] 한글 없는 리뷰 → NaN 교체: {korean_mask.sum()}행')

# ── 7. 텍스트 없는 행 제거
before = len(df)
df = df[df['review_text'].notna()].copy()
print(f'[7] 텍스트 없는 행 제거: {before} → {len(df)}행')
df['source'] = 'bbabang'

# ── 저장
df.to_csv('bbabang_reviews_final.csv', index=False, encoding='utf-8-sig')

print()
print('=== 결측치 ===')
print(df.isnull().sum())
print(f'[완료] shape: {df.shape}')