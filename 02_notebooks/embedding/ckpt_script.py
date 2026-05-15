import numpy as np
from pathlib import Path

CKPT_PATH = Path(r"D:\project\Nolit\notebooks\data\faiss_bgg_reviews_ckpt.npy")
CKPT_DIR  = Path(r"D:\project\Nolit\notebooks\data\ckpt_chunks")
CKPT_DIR.mkdir(exist_ok=True)

print("기존 체크포인트 로드 중... (시간이 걸릴 수 있어요)")
data = np.load(str(CKPT_PATH))
print(f"총 {len(data):,}개 로드 완료")

# 1000개씩 청크로 저장
CHUNK_SIZE = 1000
for i in range(0, len(data), CHUNK_SIZE):
    chunk = data[i:i+CHUNK_SIZE]
    chunk_path = CKPT_DIR / f"chunk_{i//CHUNK_SIZE:04d}.npy"
    np.save(str(chunk_path), chunk)

print(f"청크 분할 완료! {len(list(CKPT_DIR.glob('chunk_*.npy')))}개 파일 생성됨")