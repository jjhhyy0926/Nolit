from django.db import models


# =====================================================================
# 공통 추상 모델
# =====================================================================

class BaseContent(models.Model):
    """보드게임 / 방탈출 / 머더미스터리 공통 필드"""

    name        = models.CharField(max_length=200, verbose_name="게임명")
    rating      = models.FloatField(null=True, blank=True, verbose_name="평점")
    players_min = models.PositiveIntegerField(null=True, blank=True, verbose_name="최소 인원")
    players_max = models.PositiveIntegerField(null=True, blank=True, verbose_name="최대 인원")
    play_time   = models.PositiveIntegerField(null=True, blank=True, verbose_name="플레이 시간(분)")
    description = models.TextField(blank=True, verbose_name="게임 설명")
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    def __str__(self):
        return self.name

    @property
    def players_display(self):
        """'4~5명' 형태로 반환"""
        if self.players_min and self.players_max:
            if self.players_min == self.players_max:
                return f"{self.players_min}명"
            return f"{self.players_min}~{self.players_max}명"
        return ""

    @property
    def play_time_display(self):
        """'120분' 형태로 반환"""
        return f"{self.play_time}분" if self.play_time else ""


# =====================================================================
# 보드게임
# =====================================================================

class BoardGame(BaseContent):
    DIFFICULTY_CHOICES = [
        ("초급", "초급"),
        ("중급", "중급"),
        ("고급", "고급"),
    ]

    difficulty  = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, blank=True, verbose_name="난이도")
    publisher   = models.CharField(max_length=100, blank=True, verbose_name="출판사")
    designer    = models.CharField(max_length=100, blank=True, verbose_name="디자이너")
    bgg_id      = models.IntegerField(null=True, blank=True, verbose_name="BGG ID")
    bgg_rank    = models.IntegerField(null=True, blank=True, verbose_name="BGG 순위")
    category    = models.CharField(max_length=100, blank=True, verbose_name="카테고리")
    tags        = models.JSONField(default=list, blank=True, verbose_name="태그")
    name_eng    = models.CharField(max_length=200, blank=True, verbose_name="영어 이름")

    class Meta:
        db_table  = "boardgame"
        verbose_name = "보드게임"
        verbose_name_plural = "보드게임 목록"
        ordering = ["-rating"]


# =====================================================================
# 방탈출
# =====================================================================

class Escape(BaseContent):
    DIFFICULTY_CHOICES = [
        ("하", "하"),
        ("중", "중"),
        ("상", "상"),
    ]

    difficulty  = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, blank=True, verbose_name="난이도")
    region      = models.CharField(max_length=50, blank=True, verbose_name="지역")
    brand       = models.CharField(max_length=100, blank=True, verbose_name="브랜드")
    theme       = models.CharField(max_length=100, blank=True, verbose_name="테마")
    fear_level  = models.PositiveIntegerField(null=True, blank=True, verbose_name="공포도(0~5)")
    tags        = models.JSONField(default=list, blank=True, verbose_name="태그")

    class Meta:
        db_table  = "escape"
        verbose_name = "방탈출"
        verbose_name_plural = "방탈출 목록"
        ordering = ["-rating"]


# =====================================================================
# 머더미스터리 (크라임씬)
# =====================================================================

class CrimeScene(BaseContent):
    DIFFICULTY_CHOICES = [
        ("입문", "입문"),
        ("중급", "중급"),
        ("고급", "고급"),
    ]

    difficulty      = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, blank=True, verbose_name="난이도")
    series          = models.CharField(max_length=100, blank=True, verbose_name="시리즈")
    maker           = models.CharField(max_length=100, blank=True, verbose_name="제작")
    publisher       = models.CharField(max_length=100, blank=True, verbose_name="출판사")
    publisher_kr    = models.CharField(max_length=100, blank=True, verbose_name="국내 출판사")
    tags            = models.JSONField(default=list, blank=True, verbose_name="태그")

    class Meta:
        db_table  = "crimescene"
        verbose_name = "머더미스터리"
        verbose_name_plural = "머더미스터리 목록"
        ordering = ["-rating"]