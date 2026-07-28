# 식습관 기반 대사증후군 위험요인 분석 — 배포 가이드

## 구조
- `inference_service/` — Python FastAPI. LightGBM 5개 질병 모델 + SHAP. Cloudflare Workers는 Python을 못 돌리므로 별도 서버 필요.
- `frontend/` — 정적 HTML/JS 설문 폼. Cloudflare Pages로 그대로 배포 가능.
- `supabase/schema.sql` — 설문/예측 결과 저장 테이블.

## 1. 로컬에서 먼저 확인
```bash
cd webapp/inference_service
pip install -r requirements.txt
export PRODUCTION_MODEL_DIR=../../result/production_models
export PROJECT_SRC_DIR=../../src
uvicorn main:app --reload --port 8000
# http://localhost:8000/health 로 정상 응답 확인
```
`frontend/index.html`을 브라우저로 열면(또는 `python -m http.server`) `config.js`의 `API_BASE_URL=http://localhost:8000`으로 바로 테스트 가능.

## 2. 추론 서버 배포 (Cloudflare Workers는 불가 — Python/LightGBM 미지원)
Render, Railway, Fly.io 중 하나에 `inference_service/Dockerfile`로 배포 (레포 루트를 빌드 컨텍스트로):
```bash
docker build -f webapp/inference_service/Dockerfile -t diet-mets-api .
```
플랫폼에 로그인/프로젝트 생성은 각자 계정으로 진행해야 함(에이전트가 대신 로그인 불가). 배포 후 발급되는 URL을 `frontend/config.js`의 `API_BASE_URL`에 넣기.

## 3. Supabase (설문+예측 결과 저장, 선택사항)
1. https://supabase.com 에서 프로젝트 생성
2. SQL Editor에서 `supabase/schema.sql` 실행
3. Project Settings > API 에서 URL, anon public key 확인 → `frontend/config.js`의 `SUPABASE_URL`, `SUPABASE_ANON_KEY`에 입력
   (비워두면 그냥 저장을 생략하고 예측 결과만 화면에 표시함)

## 4. 프론트엔드 Cloudflare Pages 배포
```bash
wrangler login   # 최초 1회, 브라우저 인증 필요 (에이전트가 대신 못 함)
cd webapp/frontend
wrangler pages deploy . --project-name=diet-mets-screening
```

## 5. 모델 갱신
데이터가 늘어나거나 재학습이 필요하면:
```bash
cd src
python train_production_models.py
```
`result/production_models/`가 갱신되며, 추론 서버 재배포(Docker 이미지 재빌드) 필요.

## 알아둘 점
- 정확도: 5개 질병 평균 Test F1-macro ≈ 0.61 (개선/유지/악화 3-class). 스크리닝 참고용이며 진단 도구 아님.
- 검진수치(bio) 11개가 예측에 가장 큰 영향을 줌 — 사용자가 최근 검진 결과를 모르면 정확도가 크게 떨어짐.
- `frontend/diet_items.js`의 설문 문항 라벨 중 커피 외에는 정확한 원 설문 문구가 코드북에 없어 일반적인 빈도 표현으로 채워 넣었음. 실제 서비스 전 SNUH 원 설문지 문구로 교체 권장.
