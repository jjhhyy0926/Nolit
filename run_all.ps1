Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
Set-Location $PSScriptRoot
pip install -r requirements.txt -q
python manage.py embed_contents --all
Read-Host "완료! 엔터를 누르면 종료됩니다"
