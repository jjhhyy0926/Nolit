"""
recommender/management/commands/embed_contents.py

[ 사용법 ]
  python manage.py embed_contents --all
  python manage.py embed_contents --source bgg_stats
  python manage.py embed_contents --source bbabang_stats
  python manage.py embed_contents --list
"""

from django.core.management.base import BaseCommand
from recommender.rag.embeddings import run_embedding, run_all
from recommender.rag.config import get_config, list_sources


class Command(BaseCommand):
    help = "Nolit 데이터소스 임베딩 생성 및 FAISS 저장"

    def add_arguments(self, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--all",
            action="store_true",
            help="모든 데이터소스 순차 임베딩"
        )
        group.add_argument(
            "--source",
            choices=list_sources(),
            help="특정 데이터소스만 임베딩"
        )
        group.add_argument(
            "--list",
            action="store_true",
            help="사용 가능한 데이터소스 목록 출력"
        )

    def handle(self, *args, **options):

        # ── 목록 출력 ──────────────────────────
        if options["list"]:
            cfg = get_config()
            self.stdout.write("\n사용 가능한 데이터소스:\n")
            for name, src in cfg["sources"].items():
                engine = src.get("engine", "openai")
                ckpt   = "✅ ckpt" if src.get("use_ckpt") else "     "
                self.stdout.write(
                    f"  {name:<22} engine={engine:<6}  {ckpt}"
                )
            return

        # ── 전체 실행 ──────────────────────────
        if options["all"]:
            self.stdout.write(self.style.SUCCESS("\n전체 임베딩 시작\n"))
            try:
                run_all()
                self.stdout.write(self.style.SUCCESS("\n🎉 전체 임베딩 완료!"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"\n❌ 오류: {e}"))
                raise

        # ── 특정 소스 실행 ─────────────────────
        elif options["source"]:
            source = options["source"]
            try:
                run_embedding(source)
                self.stdout.write(self.style.SUCCESS(f"✅ [{source}] 완료!"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"❌ 오류: {e}"))
                raise
