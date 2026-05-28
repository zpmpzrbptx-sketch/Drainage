function fmt(v, d = 2) {
  const n = Number(v || 0);
  return Number.isFinite(n) ? n.toFixed(d) : '0.00';
}

function avg(arr = []) {
  if (!arr.length) return 0;
  return arr.reduce((s, v) => s + Number(v || 0), 0) / arr.length;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function normalizePayload(payload = {}) {
  const fallbackSeries = {
    rain: Array.isArray(payload.rain) ? payload.rain : [],
    water: Array.isArray(payload.water) ? payload.water : [[], [], []],
    energy: Array.isArray(payload.energy) ? payload.energy : [],
    storage: [[], [], []],
    actions: [],
  };
  const series = payload.series || fallbackSeries;
  return {
    ...payload,
    series: {
      ...fallbackSeries,
      ...series,
      rain: Array.isArray(series.rain) ? series.rain : fallbackSeries.rain,
      water: Array.isArray(series.water) ? series.water : fallbackSeries.water,
      energy: Array.isArray(series.energy) ? series.energy : fallbackSeries.energy,
      storage: Array.isArray(series.storage) ? series.storage : fallbackSeries.storage,
      actions: Array.isArray(series.actions) ? series.actions : fallbackSeries.actions,
    },
    kpi: payload.kpi || {
      rain_now: avg(fallbackSeries.rain),
      avg_water_now: 0,
      total_energy: avg(fallbackSeries.energy),
      risk_score: 0,
      system_score: 100,
      avg_reward: 0,
      pump_on_ratio: [0, 0, 0],
    },
    meta: payload.meta || { mode: 'rule', step: 0 },
    events: Array.isArray(payload.events) ? payload.events : [],
  };
}

function setRiskVisual(riskScore) {
  let risk = '低';
  let cls = 'low';
  if (riskScore >= 75) {
    risk = '高';
    cls = 'high';
  } else if (riskScore >= 45) {
    risk = '中';
    cls = 'medium';
  }

  const riskEl = document.getElementById('riskVal');
  if (riskEl) {
    riskEl.textContent = risk;
    riskEl.className = `number ${cls}`;
  }

  const chip = document.getElementById('riskChip');
  if (chip) {
    chip.textContent = `风险等级: ${risk}`;
    chip.className = `risk-chip ${cls}`;
  }

  return { risk, cls };
}

function updateNodeBars(series) {
  const water = Array.isArray(series.water) ? series.water : [[], [], []];
  const storage = Array.isArray(series.storage) ? series.storage : [[], [], []];

  for (let i = 0; i < 3; i += 1) {
    const w = Number(water[i]?.[water[i].length - 1] || 0);
    const s = Number(storage[i]?.[storage[i].length - 1] || 0);

    setText(`nodeWater${i + 1}Val`, `${fmt(w)}m`);
    setText(`nodeStorage${i + 1}Val`, `蓄水 ${fmt(s)}`);

    const bar = document.getElementById(`nodeWater${i + 1}Bar`);
    if (bar) {
      bar.style.width = `${Math.max(0, Math.min(100, (w / 10) * 100))}%`;
    }
  }
}

function updatePumpStatus(payload) {
  const ratio = payload.kpi?.pump_on_ratio || [0, 0, 0];
  const actions = payload.series?.actions || [];
  const lastAction = actions[actions.length - 1] || [0, 0, 0];

  for (let i = 0; i < 3; i += 1) {
    const on = Number(lastAction[i] || 0) > 0;
    const stateEl = document.getElementById(`pumpState${i + 1}`);
    if (stateEl) {
      stateEl.textContent = on ? '开启' : '关闭';
      stateEl.className = `pump-state ${on ? 'on' : 'off'}`;
    }
    setText(`pumpRatio${i + 1}`, `开启占比 ${fmt(ratio[i] || 0, 0)}%`);
  }
}

function renderInsights(payload) {
  const kpi = payload.kpi || {};
  const meta = payload.meta || {};
  const events = payload.events || [];
  const topEvent = events[0]?.message || '系统运行平稳，持续监测中。';

  const riskScore = Number(kpi.risk_score || 0);
  let actionText = '维持当前调度策略';
  if (riskScore >= 75) {
    actionText = '立即提高预排强度并优先开启泵站';
  } else if (riskScore >= 45) {
    actionText = '提前预排，关注高水位节点';
  }

  setText('modeBadge', `模式: ${(meta.mode || 'rule').toUpperCase()}`);
  setText('stepBadge', `Step ${meta.step ?? 0}`);
  setText('insightMode', (meta.mode || 'rule').toUpperCase());
  setText('insightRisk', `${fmt(riskScore, 0)}/100`);
  setText('insightAction', actionText);
  setText('insightExplain', topEvent);

  const snapshot = `当前平均水位 ${fmt(kpi.avg_water_now)} m，累计能耗 ${fmt(kpi.total_energy)} kWh。`;
  const hint = `最近告警：${topEvent}`;
  setText('snapshotText', snapshot);
  setText('snapshotHint', hint);
}

function renderKpis(payload) {
  const kpi = payload.kpi || {};
  const riskScore = Number(kpi.risk_score || 0);

  setText('rainVal', fmt(kpi.rain_now));
  setText('waterVal', fmt(kpi.avg_water_now));
  setText('energyVal', fmt(kpi.total_energy));
  setText('systemScoreVal', fmt(kpi.system_score, 0));
  setText('avgRewardVal', fmt(kpi.avg_reward));
  setText('riskScoreVal', `${fmt(riskScore, 0)} / 100`);

  setRiskVisual(riskScore);
}

async function loadData() {
  try {
    const res = await fetch('/api/data?steps=20');
    const raw = await res.json();
    const payload = normalizePayload(raw);

    renderKpis(payload);
    updateNodeBars(payload.series);
    updatePumpStatus(payload);
    renderInsights(payload);
  } catch (err) {
    setText('snapshotText', '数据加载失败');
    setText('snapshotHint', '请检查后端服务是否正常运行。');
  }
}

setInterval(() => {
  setText('time', new Date().toLocaleString());
}, 1000);

loadData();
setInterval(loadData, 3000);
