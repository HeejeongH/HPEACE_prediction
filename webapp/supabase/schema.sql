-- 설문 응답 + 예측 결과 저장 테이블. 개인식별정보(이름/연락처 등)는 수집하지 않음.
create table if not exists survey_predictions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  survey jsonb not null,      -- 사용자가 입력한 설문(성별/나이/신장/흡연/활동량/음주/bio/diet)
  prediction jsonb not null   -- FastAPI /predict 응답 전체
);

alter table survey_predictions enable row level security;

-- 익명 사용자가 자기 응답을 저장(insert)만 할 수 있게 허용. 조회/수정/삭제는 막아 둠(관리자만 대시보드에서 확인).
create policy "anon can insert" on survey_predictions
  for insert to anon
  with check (true);
