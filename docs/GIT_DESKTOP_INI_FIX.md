# Git Desktop.ini 오류 해결 가이드

## 🔴 오류 증상

```
fatal: bad object refs/desktop.ini
error: https://github.com/... did not send all necessary objects
```

Git pull 실행 시 위와 같은 오류가 발생합니다.

---

## 🔍 원인

Windows에서 폴더를 열 때 자동으로 생성되는 `desktop.ini` 파일이 `.git/refs/` 폴더에 들어가서 Git 저장소를 손상시켰습니다.

---

## 🔧 해결 방법

### ✅ 방법 1: 간단한 수정 (먼저 시도)

```bash
# 1. 문제가 되는 desktop.ini 파일 삭제
del /F /A:H .git\refs\desktop.ini

# 2. 다시 pull 시도
git pull origin main
```

**성공하면** → 완료! 🎉  
**실패하면** → 방법 2로 이동

---

### ✅ 방법 2: 강제 리셋 (권장)

```bash
# 1. 원격 저장소 최신 정보 가져오기
git fetch origin

# 2. 로컬을 원격과 동일하게 강제 리셋
git reset --hard origin/main

# 3. 확인
git status
```

**장점**: 빠르고 확실함  
**주의**: 로컬의 커밋되지 않은 변경사항이 사라짐 (data 폴더는 안전)

**성공하면** → 완료! 🎉  
**실패하면** → 방법 3으로 이동

---

### ✅ 방법 3: .git 폴더 전체 청소

```bash
# 1. .git 폴더의 모든 desktop.ini 삭제
cd .git
del /S /A:H desktop.ini
cd ..

# 2. Git garbage collection 실행
git gc --prune=now

# 3. 원격 저장소 다시 가져오기
git fetch origin

# 4. 강제 리셋
git reset --hard origin/main
```

**성공하면** → 완료! 🎉  
**실패하면** → 방법 4로 이동

---

### ✅ 방법 4: 저장소 다시 클론 (최후의 수단)

⚠️ **주의**: 이 방법을 사용하기 전에 `data/` 폴더를 백업하세요!

```bash
# 0. 현재 위치 확인 (프로젝트 폴더 안에 있어야 함)
cd "G:\내 드라이브\#인력양성\3. 식이_SNUH\#Prediction"

# 1. 부모 폴더로 이동
cd ..

# 2. 현재 폴더 이름 변경 (백업)
ren "#Prediction" "#Prediction_backup"

# 3. 저장소 새로 클론
git clone https://github.com/HeejeongH/HPEACE_prediction.git "#Prediction"

# 4. 새 폴더로 이동
cd "#Prediction"

# 5. 백업한 data 폴더 복사 (데이터 파일 보존)
xcopy "..\#Prediction_backup\data" "data" /E /I /Y

# 6. 확인
dir data
git status
```

성공하면:
```bash
# 7. 백업 폴더 삭제 (선택사항)
cd ..
rmdir /S /Q "#Prediction_backup"
```

---

## 🛡️ 재발 방지

이미 `.gitignore`에 `desktop.ini`가 추가되어 있으므로, 앞으로는 이런 문제가 발생하지 않습니다.

현재 `.gitignore` 내용:
```gitignore
# Windows
desktop.ini
Thumbs.db
*.lnk

# macOS
.DS_Store
.AppleDouble
.LSOverride
```

---

## 📋 단계별 체크리스트

순서대로 시도하고, 성공하면 다음 단계는 건너뛰세요:

- [ ] **1단계**: `del /F /A:H .git\refs\desktop.ini` → `git pull origin main`
- [ ] **2단계**: `git fetch origin` → `git reset --hard origin/main`
- [ ] **3단계**: `.git` 폴더 청소 → `git gc --prune=now` → 다시 pull
- [ ] **4단계**: 저장소 다시 클론 (data 폴더 백업 필수!)

---

## ❓ 자주 묻는 질문

### Q: 로컬에서 수정한 코드가 사라지나요?
**A**: 방법 2, 3, 4는 커밋되지 않은 변경사항을 잃을 수 있습니다. 하지만 `data/` 폴더는 `.gitignore`에 있어서 안전합니다.

### Q: 왜 desktop.ini가 생기나요?
**A**: Windows가 폴더를 열 때 자동으로 생성합니다. 이제 `.gitignore`에 추가되어서 더 이상 문제없습니다.

### Q: 가장 빠른 방법은?
**A**: 방법 2 (강제 리셋)가 가장 빠르고 확실합니다.

### Q: 가장 안전한 방법은?
**A**: 방법 1부터 순서대로 시도하는 것이 가장 안전합니다.

---

## 🆘 여전히 안 되면?

모든 방법을 시도했는데도 안 된다면:

```bash
# Git 저장소 상태 확인
git fsck --full

# 오류 메시지 복사해서 검색
```

또는 새 폴더에서 다시 시작:
```bash
# 완전히 새로운 위치에 클론
cd "G:\내 드라이브\#인력양성\3. 식이_SNUH"
git clone https://github.com/HeejeongH/HPEACE_prediction.git "#Prediction_new"

# 데이터 파일만 복사
xcopy "#Prediction\data" "#Prediction_new\data" /E /I /Y
```

---

## ✅ 해결 후 확인사항

문제가 해결되었다면:

```bash
# 1. Git 상태 확인
git status

# 2. 최신 코드 확인
git log --oneline -5

# 3. 실행 테스트
python run_training.py
```

모두 정상이면 완료입니다! 🎉
