#!/bin/bash
################################################################################
# Desktop.ini 문제 자동 해결 스크립트 (Git Bash용)
################################################################################

echo ""
echo "================================================================================"
echo "🔧 Desktop.ini 문제 해결 스크립트"
echo "================================================================================"
echo ""

# 현재 디렉토리 확인
echo "📂 현재 위치: $(pwd)"
echo ""

# 1. Git에서 desktop.ini 완전 제거
echo "================================================================================"
echo "📝 Step 1: Git에서 desktop.ini 제거"
echo "================================================================================"
git rm -f --cached desktop.ini 2>/dev/null || true
git rm -f --cached "**/desktop.ini" 2>/dev/null || true
git rm -f --cached .git/refs/desktop.ini 2>/dev/null || true
echo "✅ Git 캐시에서 desktop.ini 제거 완료"
echo ""

# 2. 로컬 desktop.ini 파일 삭제
echo "================================================================================"
echo "📝 Step 2: 로컬 desktop.ini 파일 삭제"
echo "================================================================================"
find . -type f -name "desktop.ini" -delete 2>/dev/null || true
find . -type f -name "Thumbs.db" -delete 2>/dev/null || true
echo "✅ 로컬 desktop.ini 파일 삭제 완료"
echo ""

# 3. .git/refs/desktop.ini 특별 처리
echo "================================================================================"
echo "📝 Step 3: .git/refs/desktop.ini 제거"
echo "================================================================================"
if [ -f ".git/refs/desktop.ini" ]; then
    rm -f .git/refs/desktop.ini
    echo "✅ .git/refs/desktop.ini 삭제 완료"
else
    echo "ℹ️  .git/refs/desktop.ini 파일 없음"
fi
echo ""

# 4. Git 정리
echo "================================================================================"
echo "📝 Step 4: Git 저장소 정리"
echo "================================================================================"
git gc --prune=now --aggressive
echo "✅ Git 저장소 정리 완료"
echo ""

# 5. .gitignore 확인
echo "================================================================================"
echo "📝 Step 5: .gitignore 확인"
echo "================================================================================"
if grep -q "desktop.ini" .gitignore 2>/dev/null; then
    echo "✅ .gitignore에 desktop.ini 이미 존재"
else
    echo "desktop.ini" >> .gitignore
    echo "✅ .gitignore에 desktop.ini 추가 완료"
fi
echo ""

# 6. 변경사항 커밋 (있는 경우)
echo "================================================================================"
echo "📝 Step 6: 변경사항 확인"
echo "================================================================================"
if git status --porcelain | grep -q "desktop.ini"; then
    echo "📤 변경사항 커밋 중..."
    git add .gitignore
    git commit -m "Fix: Remove desktop.ini from Git tracking"
    echo "✅ 변경사항 커밋 완료"
else
    echo "ℹ️  커밋할 변경사항 없음"
fi
echo ""

# 7. 원격 저장소와 동기화
echo "================================================================================"
echo "📝 Step 7: 원격 저장소와 동기화"
echo "================================================================================"
echo "🔄 원격 저장소에서 최신 코드 받기..."
git fetch origin
git reset --hard origin/main
echo "✅ 동기화 완료"
echo ""

echo "================================================================================"
echo "✅ 모든 작업 완료!"
echo "================================================================================"
echo ""
echo "💡 앞으로 이 문제 방지하기:"
echo "   1. Windows 탐색기에서 이 폴더를 열 때 .git 폴더를 건드리지 마세요"
echo "   2. Git 명령어는 항상 명령 프롬프트나 Git Bash에서 실행하세요"
echo "   3. 이 스크립트를 주기적으로 실행하세요"
echo ""
