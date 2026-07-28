// 설문 폼 렌더링 + 제출 + 결과 표시 + (선택) Supabase 저장

const dietList = document.getElementById('diet-list');
DIET_ITEMS.forEach((item) => {
  const div = document.createElement('div');
  div.className = 'field';
  const options = item.options.map((label, idx) =>
    `<option value="${idx + 1}">${label}</option>`
  ).join('');
  div.innerHTML = `<label>${item.question}</label>
    <select name="diet.${item.key}" required>${options}</select>`;
  dietList.appendChild(div);
});

function formToPayload(form) {
  const fd = new FormData(form);
  const payload = { bio: {}, diet: {} };
  for (const [key, value] of fd.entries()) {
    if (key.startsWith('bio.')) {
      payload.bio[key.slice(4)] = parseFloat(value);
    } else if (key.startsWith('diet.')) {
      payload.diet[key.slice(5)] = parseInt(value, 10);
    } else if (key === '나이' || key === '신장' || key === '흡연' || key === '활동량' || key === '음주') {
      payload[key] = key === '나이' || key === '신장' ? parseFloat(value) : parseInt(value, 10);
    } else {
      payload[key] = value;
    }
  }
  return payload;
}

function riskClass(predicted) {
  if (predicted === '악화') return 'risk-worsen';
  if (predicted === '개선') return 'risk-improve';
  return 'risk-maintain';
}

function renderResults(data) {
  const box = document.getElementById('results');
  box.innerHTML = '<h2 style="margin:1.5rem 0 1rem">분석 결과</h2>';
  data.results.forEach((r) => {
    const card = document.createElement('div');
    card.className = `disease-card ${riskClass(r.predicted)}`;
    const pct = (x) => Math.round(x * 100);
    const factorsHtml = r.top_diet_factors
      .map((f) => `${f.feature} (${f.shap_contribution > 0 ? '위험도↑' : '위험도↓'})`)
      .join(', ');
    card.innerHTML = `
      <h3>${r.disease_kr} — 예측: ${r.predicted}</h3>
      <div class="bar-row"><span>개선 ${pct(r.prob_improve)}%</span><div class="bar-track"><div class="bar-fill" style="width:${pct(r.prob_improve)}%"></div></div></div>
      <div class="bar-row"><span>유지 ${pct(r.prob_maintain)}%</span><div class="bar-track"><div class="bar-fill" style="width:${pct(r.prob_maintain)}%"></div></div></div>
      <div class="bar-row"><span>악화 ${pct(r.prob_worsen)}%</span><div class="bar-track"><div class="bar-fill" style="width:${pct(r.prob_worsen)}%"></div></div></div>
      <div class="factors"><b>주요 관련 식습관</b>${factorsHtml}</div>
    `;
    box.appendChild(card);
  });
  box.style.display = 'block';
  box.scrollIntoView({ behavior: 'smooth' });
}

async function saveToSupabase(payload, response) {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return; // 미설정 시 저장 생략
  try {
    await fetch(`${SUPABASE_URL}/rest/v1/survey_predictions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        apikey: SUPABASE_ANON_KEY,
        Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
      },
      body: JSON.stringify({ survey: payload, prediction: response }),
    });
  } catch (e) {
    console.warn('Supabase 저장 실패(예측 결과 표시에는 영향 없음):', e);
  }
}

document.getElementById('surveyForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById('submitBtn');
  const errorBox = document.getElementById('errorBox');
  errorBox.style.display = 'none';
  btn.disabled = true;
  btn.textContent = '분석 중...';

  const payload = formToPayload(form);

  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `서버 오류 (${res.status})`);
    }
    const data = await res.json();
    renderResults(data);
    saveToSupabase(payload, data);
  } catch (err) {
    errorBox.textContent = `분석 요청 실패: ${err.message}`;
    errorBox.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = '위험도 분석하기';
  }
});
