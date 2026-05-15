import pandas as pd
import numpy as np
import re

df = pd.read_csv('bbabang_stats.csv')

score_cols = ['difficulty', 'horror', 'activity', 'satisfaction', 'puzzle', 'story', 'interior', 'production']

# ── 1. 모든 점수 컬럼 -1인 행 제거
before = len(df)
df = df[~(df[score_cols] == -1).all(axis=1)]
print(f'[1] -1 행 제거: {before} → {len(df)}')

# ── 2. 부분 -1 → NaN 처리
for col in score_cols:
    df[col] = df[col].replace(-1, np.nan)
print('[2] 부분 -1 → NaN 처리 완료')
for col in score_cols:
    print(f'    {col}: NaN {df[col].isna().sum()}개')

# ── 3. player_count → max_players rename, 이상치 처리
df = df.rename(columns={'player_count': 'max_players'})
df.loc[df['max_players'] == 0, 'max_players'] = np.nan
df['max_players'] = df['max_players'].astype('Int64')

# ── 4. description 클렌징
df['description'] = (
    df['description']
    .fillna('')
    .str.replace(r'\r\n|\r|\n', ' ', regex=True)
    .str.strip()
)

# ── 5. price int 변환
df['price'] = df['price'].astype('Int64')

# ── 6. address → area(시/도), location(시/군/구) 추출
area_map = {
    '강원': '강원', '강원도': '강원', '강원특별자치도': '강원',
    '경기': '경기', '경기도': '경기',
    '인천': '인천', '인천광역시': '인천',
}

def extract_area(addr):
    if pd.isna(addr):
        return np.nan
    first = str(addr).split()[0]
    return area_map.get(first, first)

def extract_location(addr):
    if pd.isna(addr):
        return np.nan
    parts = str(addr).split()
    return parts[1] if len(parts) >= 2 else np.nan

df['area'] = df['address'].apply(extract_area)
df['location'] = df['address'].apply(extract_location)

# ── 7. interior / production 0~5 스케일 정규화
INTERIOR_MAX = 6.0
PRODUCTION_MAX = 6.5

df['interior']   = (df['interior']   / INTERIOR_MAX   * 5).round(2)
df['production'] = (df['production'] / PRODUCTION_MAX * 5).round(2)

print()
print('[7] 스케일 정규화 완료')
print(f'    interior:   0~{INTERIOR_MAX} → 0~5 (÷{INTERIOR_MAX}×5)')
print(f'    production: 0~{PRODUCTION_MAX} → 0~5 (÷{PRODUCTION_MAX}×5)')

# ── 8. 컬럼 순서 정리
col_order = [
    'title', 'store_name', 'area', 'location',
    'description', 'playing_time', 'max_players', 'price',
    'difficulty', 'horror', 'activity',
    'satisfaction', 'puzzle', 'story', 'interior', 'production'
]
df = df[col_order]
df['source'] = 'bbabang'

# ── 저장
df.to_csv('bbabang_stats_final.csv', index=False, encoding='utf-8-sig')

print()
print(f'[완료] shape: {df.shape}')
print()
print('=== 결측치 ===')
print(df.isnull().sum())
print()
print('=== area 분포 ===')
print(df['area'].value_counts())
print()
print('=== 점수 컬럼 범위 ===')
print(df[score_cols].describe().round(2))