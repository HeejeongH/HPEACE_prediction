@echo off
echo ========================================
echo Git Desktop.ini 문제 해결 스크립트
echo ========================================
echo.

echo Step 1: .git/refs 폴더의 Desktop.ini 삭제
cd .git\refs
if exist desktop.ini del /F /A desktop.ini
if exist Desktop.ini del /F /A Desktop.ini
cd ..\..

echo Step 2: Git 원격 상태 확인
git fetch origin --prune

echo Step 3: 로컬 변경사항 백업 (있다면)
git status

echo.
echo Step 4: 강제로 원격과 동기화
git reset --hard origin/main

echo Step 5: 정리
git clean -fd

echo.
echo ========================================
echo 완료! 이제 정상적으로 pull 가능합니다.
echo ========================================
pause
