@echo off
cd /d %~dp0
call conda activate aistudy_env
pip install -r requirements.txt -q
python manage.py embed_contents --all
pause
