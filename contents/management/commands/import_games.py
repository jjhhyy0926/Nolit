import os
import json
import math
import time
from django.core.management.base import BaseCommand
from contents.models import BoardGame, Escape, CrimeScene
from django.conf import settings

VECTORSTORE = os.path.join(settings.BASE_DIR, "04_vectorstore")


class Command(BaseCommand):
    help = "meta JSON 파일을 읽어 MySQL DB에 저장합니다"

    def handle(self, *args, **kwargs):
        total_start = time.time()
        self.stdout.write("=== 게임 데이터 import 시작 ===\n")

        self._import_boardgame_boardlife()
        self._import_boardgame_bgg()
        self._import_escape_bbabang()
        self._import_crimescene_murdermysterylog()
        self._import_crimescene_murmynow()

        elapsed = time.time() - total_start
        self.stdout.write(self.style.SUCCESS(f"\n=== 완료 ({elapsed:.1f}초) ==="))
        self.stdout.write(f"보드게임: {BoardGame.objects.count()}개")
        self.stdout.write(f"방탈출:   {Escape.objects.count()}개")
        self.stdout.write(f"크라임씬: {CrimeScene.objects.count()}개")

    # ── 보드게임: boardlife ─────────────────────────────────
    def _import_boardgame_boardlife(self):
        stats_path   = os.path.join(VECTORSTORE, "faiss_boardlife_stats_meta.json")
        reviews_path = os.path.join(VECTORSTORE, "faiss_boardlife_reviews_meta.json")

        if not os.path.exists(stats_path):
            self.stdout.write("[SKIP] boardlife stats 없음")
            return

        self.stdout.write("[boardlife] 보드게임 import 중...")
        start = time.time()

        # 리뷰 평점 맵 (title → [rating, ...])
        reviews_map = {}
        if os.path.exists(reviews_path):
            for item in self._load(reviews_path):
                title  = item.get("title", "")
                rating = item.get("rating")
                if title and rating is not None:
                    reviews_map.setdefault(title, []).append(self._to_float(rating))

        data  = self._load(stats_path)
        total = len(data)
        count = 0

        for i, item in enumerate(data):
            name = self._to_str(item.get("title"))
            if not name:
                continue
            if i % 300 == 0:
                self.stdout.write(f"  {i}/{total} ({time.time()-start:.1f}초)")

            min_p = self._to_int(item.get("min_players"))
            max_p = self._to_int(item.get("max_players"))
            # boardlife: min_time / max_time
            min_t = self._to_int(item.get("min_time"))
            max_t = self._to_int(item.get("max_time"))
            if min_t and max_t:
                play_time = int((min_t + max_t) / 2)
            else:
                play_time = min_t or None

            # 메커니즘: 문자열 ("|" 구분)
            mechanism_raw = item.get("mechanism", "") or ""
            if isinstance(mechanism_raw, list):
                mechanism_raw = " | ".join(mechanism_raw)
            elif not isinstance(mechanism_raw, str):
                mechanism_raw = str(mechanism_raw)
            tags = [m.strip() for m in mechanism_raw.split("|") if m.strip()][:6]

            # designer: 문자열
            designer = item.get("designer", "") or ""
            if isinstance(designer, list):
                designer = " | ".join(designer)

            # 별점: boardlife avg_rating 은 10점 만점 → 5점으로 정규화
            raw_rating = self._to_float(item.get("avg_rating"))
            rating = round(raw_rating / 2, 1) if raw_rating else None

            BoardGame.objects.update_or_create(
                name=name,
                defaults={
                    "rating":      rating,
                    "players_min": min_p,
                    "players_max": max_p,
                    "play_time":   play_time,
                    "description": "",
                    "image_url":   self._to_str(item.get("image")),
                    "difficulty":  self._weight_to_difficulty(item.get("weight")),
                    "designer":    self._to_str(designer),
                    "mechanism":   mechanism_raw[:200],
                    "category":    self._to_str(item.get("type")),
                    "tags":        tags,
                    "reviews":     [],
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"[boardlife] {count}개 완료 ({time.time()-start:.1f}초)"))

    # ── 보드게임: BGG ───────────────────────────────────────
    def _import_boardgame_bgg(self):
        stats_path = os.path.join(VECTORSTORE, "faiss_bgg_stats_meta.json")
        if not os.path.exists(stats_path):
            self.stdout.write("[SKIP] bgg stats 없음")
            return

        self.stdout.write("[bgg] 보드게임 import 중...")
        start = time.time()

        data  = self._load(stats_path)
        total = len(data)
        count = 0

        for i, item in enumerate(data):
            name = self._to_str(item.get("title"))
            if not name:
                continue
            if i % 500 == 0:
                self.stdout.write(f"  {i}/{total} ({time.time()-start:.1f}초)")

            # boardlife 에 이미 있으면 스킵
            if BoardGame.objects.filter(name=name).exists():
                continue

            # mechanism: 리스트
            mechanism_list = item.get("mechanism", []) or []
            if isinstance(mechanism_list, list):
                mechanism = " | ".join(mechanism_list)
                tags = mechanism_list[:6]
            else:
                mechanism = self._to_str(mechanism_list)
                tags = []

            # designer: 리스트
            designer_raw = item.get("designer", []) or []
            if isinstance(designer_raw, list):
                designer = " | ".join(designer_raw)
            else:
                designer = self._to_str(designer_raw)

            # type: 리스트
            type_raw = item.get("type", []) or []
            if isinstance(type_raw, list):
                category = " | ".join(type_raw)
            else:
                category = self._to_str(type_raw)

            # playing_time: BGG는 단일 필드
            play_time = self._to_int(item.get("playing_time"))

            # 별점: BGG avg_rating 은 10점 만점 → 5점으로 정규화
            raw_rating = self._to_float(item.get("avg_rating"))
            rating = round(raw_rating / 2, 1) if raw_rating else None

            BoardGame.objects.update_or_create(
                name=name,
                defaults={
                    "rating":      rating,
                    "players_min": self._to_int(item.get("min_players")),
                    "players_max": self._to_int(item.get("max_players")),
                    "play_time":   play_time,
                    "description": "",
                    "image_url":   self._to_str(item.get("image")),
                    "difficulty":  self._weight_to_difficulty(item.get("weight")),
                    "designer":    designer,
                    "bgg_rank":    self._to_int(item.get("rank")),
                    "mechanism":   mechanism[:200],
                    "category":    category,
                    "tags":        tags,
                    "reviews":     [],
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"[bgg] {count}개 완료 ({time.time()-start:.1f}초)"))

    # ── 방탈출: 빠른방탈출 ─────────────────────────────────
    def _import_escape_bbabang(self):
        stats_path = os.path.join(VECTORSTORE, "faiss_bbabang_stats_metadata.json")
        if not os.path.exists(stats_path):
            stats_path = os.path.join(VECTORSTORE, "bbabang_stats_openai_metadata.json")
        if not os.path.exists(stats_path):
            self.stdout.write("[SKIP] bbabang stats 없음")
            return

        reviews_path = os.path.join(VECTORSTORE, "faiss_bbabang_reviews_metadata.json")
        if not os.path.exists(reviews_path):
            reviews_path = os.path.join(VECTORSTORE, "bbabang_reviews_openai_metadata.json")

        self.stdout.write("[bbabang] 방탈출 import 중...")
        start = time.time()

        reviews_map = {}
        if os.path.exists(reviews_path):
            for item in self._load(reviews_path):
                key = f"{item.get('title','')}|{item.get('store_name','')}"
                doc = item.get("document", "")
                if doc:
                    reviews_map.setdefault(key, []).append(doc)

        data  = self._load(stats_path)
        total = len(data)
        count = 0

        for i, item in enumerate(data):
            name       = self._to_str(item.get("title"))
            store_name = self._to_str(item.get("store_name"))
            if not name:
                continue
            if i % 100 == 0:
                self.stdout.write(f"  {i}/{total} ({time.time()-start:.1f}초)")

            horror     = item.get("horror")
            fear_level = self._to_int(float(horror) * 5) if horror is not None and not (isinstance(horror, float) and math.isnan(horror)) else None
            difficulty = self._num_to_difficulty(item.get("difficulty"))
            players_max = self._to_int(item.get("max_players"))
            play_time   = self._to_int(item.get("playing_time"))

            # satisfaction: 0~5 만점 그대로 사용
            rating = self._to_float(item.get("satisfaction"))

            tags = []
            if self._to_float(item.get("story"), 0) >= 2:   tags.append("스토리")
            if self._to_float(item.get("puzzle"), 0) >= 2:   tags.append("퍼즐")
            if self._to_float(item.get("activity"), 0) >= 0.5: tags.append("활동형")
            if fear_level is not None:
                tags.append("공포 없음" if fear_level <= 1 else "공포 있음" if fear_level >= 3 else "")

            key       = f"{name}|{store_name}"
            reviews   = reviews_map.get(key, [])[:3]
            full_name = f"{name} ({store_name})" if store_name else name

            Escape.objects.update_or_create(
                name=full_name,
                defaults={
                    "rating":      rating,
                    "players_min": 1,
                    "players_max": players_max,
                    "play_time":   play_time,
                    "description": "",
                    "difficulty":  difficulty,
                    "region":      self._to_str(item.get("location")),
                    "brand":       store_name,
                    "fear_level":  fear_level,
                    "tags":        [t for t in tags if t],
                    "reviews":     reviews,
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"[bbabang] {count}개 완료 ({time.time()-start:.1f}초)"))

    # ── 크라임씬: murdermysterylog ──────────────────────────
    def _import_crimescene_murdermysterylog(self):
        path = os.path.join(VECTORSTORE, "faiss_murdermysterylog_meta.json")
        if not os.path.exists(path):
            self.stdout.write("[SKIP] murdermysterylog 없음")
            return

        self.stdout.write("[murdermysterylog] 크라임씬 import 중...")
        start = time.time()

        data  = self._load(path)
        total = len(data)
        count = 0

        for i, item in enumerate(data):
            name = self._to_str(item.get("name"))
            if not name:
                continue
            if i % 50 == 0:
                self.stdout.write(f"  {i}/{total} ({time.time()-start:.1f}초)")

            # 리뷰: "리뷰1 || 리뷰2" 형태
            reviews_raw = item.get("reviews", "") or ""
            if isinstance(reviews_raw, str) and "||" in reviews_raw:
                reviews = [r.strip() for r in reviews_raw.split("||") if r.strip()][:3]
            elif isinstance(reviews_raw, list):
                reviews = [self._to_str(r) for r in reviews_raw[:3]]
            else:
                rev = self._to_str(reviews_raw)
                reviews = [rev] if rev else []

            # rating: 5점 만점 그대로
            rating = self._to_float(item.get("rating"))

            CrimeScene.objects.update_or_create(
                name=name,
                defaults={
                    "rating":      rating,
                    "players_min": self._to_int(item.get("min_players")),
                    "players_max": self._to_int(item.get("max_players")),
                    "play_time":   self._to_int(item.get("play_time")),
                    "description": self._to_str(item.get("description")),
                    "series":      self._to_str(item.get("시리즈")),
                    "maker":       self._to_str(item.get("제작")),
                    "tags":        [],
                    "reviews":     reviews,
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"[murdermysterylog] {count}개 완료 ({time.time()-start:.1f}초)"))

    # ── 크라임씬: murmynow ──────────────────────────────────
    def _import_crimescene_murmynow(self):
        path = os.path.join(VECTORSTORE, "faiss_murmynow_meta.json")
        if not os.path.exists(path):
            self.stdout.write("[SKIP] murmynow 없음")
            return

        self.stdout.write("[murmynow] 크라임씬 import 중...")
        start = time.time()

        data  = self._load(path)
        total = len(data)
        count = 0

        for i, item in enumerate(data):
            name = self._to_str(item.get("name"))
            if not name:
                continue
            if i % 50 == 0:
                self.stdout.write(f"  {i}/{total} ({time.time()-start:.1f}초)")

            if CrimeScene.objects.filter(name=name).exists():
                continue

            # rating: 5점 만점 그대로
            rating = self._to_float(item.get("rating"))

            CrimeScene.objects.update_or_create(
                name=name,
                defaults={
                    "rating":      rating,
                    "players_min": self._to_int(item.get("min_players")),
                    "players_max": self._to_int(item.get("max_players")),
                    "play_time":   self._to_int(item.get("min_time")),
                    "description": "",
                    "image_url":   self._to_str(item.get("image_url")),
                    "difficulty":  self._num_to_difficulty_5(item.get("difficulty")),
                    "maker":       self._to_str(item.get("author")),
                    "publisher":   self._to_str(item.get("publisher")),
                    "tags":        [],
                    "reviews":     [],
                },
            )
            count += 1

        self.stdout.write(self.style.SUCCESS(f"[murmynow] {count}개 완료 ({time.time()-start:.1f}초)"))

    # ── 유틸 ───────────────────────────────────────────────
    def _load(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.stdout.write(f"[ERROR] {path}: {e}")
            return []

    def _to_str(self, val):
        if val is None: return ""
        if isinstance(val, float) and math.isnan(val): return ""
        if isinstance(val, list): return " | ".join(str(v) for v in val)
        return str(val).strip()

    def _to_int(self, val):
        try:
            if val is None: return None
            f = float(val)
            if math.isnan(f): return None
            return int(f)
        except (TypeError, ValueError):
            return None

    def _to_float(self, val, default=None):
        try:
            if val is None: return default
            f = float(val)
            if math.isnan(f): return default
            return f
        except (TypeError, ValueError):
            return default

    def _weight_to_difficulty(self, weight):
        w = self._to_float(weight)
        if w is None: return ""
        if w < 2.0:   return "초급"
        if w < 3.0:   return "중급"
        return "고급"

    def _num_to_difficulty(self, num):
        n = self._to_float(num)
        if n is None: return ""
        if n < 2.0:   return "하"
        if n < 3.5:   return "중"
        return "상"

    def _num_to_difficulty_5(self, num):
        n = self._to_float(num)
        if n is None: return ""
        if n <= 2.0:  return "입문"
        if n <= 3.5:  return "중급"
        return "고급"