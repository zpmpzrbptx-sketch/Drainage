const state = {
  sessionId: null,
  mode: "rl",
  manualAction: [0, 0, 0],
  running: false,
  interval: null,
  history: [],
  compareChart: null,
  timelineChart: null,
};

const $ = (id) => document.getElementById(id);

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function prettyActionSource(source) {
  const key = String(source || "").toLowerCase();
  if (key === "rl") return "RL模型";
  if (key === "rule") return "规则策略";
  if (key === "rule_fallback") return "规则回退";
  if (key === "manual") return "手动输入";
  if (key === "reset") return "重置状态";
  if (key === "none") return "无动作";
  return source || "-";
}

function prettyStrategy(mode, source) {
  const sourceKey = String(source || "").toLowerCase();
  const modeKey = String(mode || "").toLowerCase();
  if (sourceKey === "rl") return "强化学习";
  if (sourceKey === "rule_fallback") return "强化学习(规则回退)";
  if (sourceKey === "manual") return "手动模式";
  if (modeKey === "rule" || sourceKey === "rule") return "规则调度";
  return mode || source || "-";
}

function setWater(id, value, maxValue = 10) {
  const el = $(id);
  if (!el) return;
  const safeMax = Number(maxValue) > 0 ? Number(maxValue) : 10;
  el.style.height = `${clamp01(Number(value || 0) / safeMax) * 100}%`;
}

function setActiveMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  toggleManualPanel();
}

function toggleManualPanel() {
  const panel = $("manualPanel");
  if (!panel) return;
  panel.style.display = state.mode === "manual" ? "block" : "none";
}

function setPumpState(action = [0, 0, 0]) {
  const arr = Array.isArray(action) ? action : [0, 0, 0];
  for (let i = 0; i < 3; i += 1) {
    const stateEl = $(`pumpState${i + 1}`);
    const cardEl = $(`pumpCard${i + 1}`);
    const on = Number(arr[i] || 0) > 0;
    if (stateEl) {
      stateEl.textContent = on ? "开启" : "关闭";
      stateEl.classList.toggle("on", on);
      stateEl.classList.toggle("off", !on);
    }
    if (cardEl) {
      cardEl.classList.toggle("on", on);
    }
  }
}

function syncManualButtons() {
  document.querySelectorAll(".manual-btn[data-pump]").forEach((btn) => {
    const idx = Number(btn.dataset.pump || -1);
    const on = idx >= 0 ? Number(state.manualAction[idx] || 0) > 0 : false;
    btn.classList.toggle("active", on);
  });
}

function normalizeAction(action) {
  const arr = Array.isArray(action) ? action : [0, 0, 0];
  return [0, 1, 2].map((i) => (Number(arr[i] || 0) > 0 ? 1 : 0));
}

async function initSession() {
  const res = await fetch("/api/sim/init", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: state.mode, max_steps: 200, seed: Date.now() % 100000 }),
  });
  const data = await res.json();
  state.sessionId = data.session_id;
  state.history = [];
  updateReplay();
  return data;
}

function renderCurrent(data) {
  setText("rain", data.rain.toFixed ? data.rain.toFixed(2) : data.rain);
  setText("stepNo", data.step ?? 0);
  setText("modelName", data.model_name || "-");
  setText("sceneName", data.scene_name || "-");
  setText("strategy", prettyStrategy(data.mode || state.mode, data.action_source));
  setText("actionSource", prettyActionSource(data.action_source));
  setText("action", `[${(data.action || []).join(", ")}]`);
  setText("risk", data.risk || "-");
  setText("node1", `${(data.water_levels?.[0] ?? 0).toFixed(2)}m`);
  setText("node2", `${(data.water_levels?.[1] ?? 0).toFixed(2)}m`);
  setText("node3", `${(data.water_levels?.[2] ?? 0).toFixed(2)}m`);
  setText("storageText1", `${(data.storage_levels?.[0] ?? 0).toFixed(2)}m`);
  setText("storageText2", `${(data.storage_levels?.[1] ?? 0).toFixed(2)}m`);
  setText("storageText3", `${(data.storage_levels?.[2] ?? 0).toFixed(2)}m`);
  setText("flow", `${(data.flow_levels?.[0] ?? 0).toFixed(2)} / ${(data.flow_levels?.[1] ?? 0).toFixed(2)} / ${(data.flow_levels?.[2] ?? 0).toFixed(2)} m³/s`);
  setText("energy", `${(data.energy ?? 0).toFixed(2)} kWh`);
  setText("overflow", `${(data.overflow ?? 0).toFixed(2)} m³`);
  setText("reward", `${(data.reward ?? 0).toFixed(2)}`);
  setText("explain", data.explain || "");

  setWater("storage1", data.storage_levels?.[0] ?? 0, 100);
  setWater("storage2", data.storage_levels?.[1] ?? 0, 100);
  setWater("storage3", data.storage_levels?.[2] ?? 0, 100);
  setPumpState(data.action || [0, 0, 0]);

  if (state.mode === "manual" || data.action_source === "manual") {
    state.manualAction = normalizeAction(data.action);
    syncManualButtons();
  }

  const pump = $("pump");
  if (pump) {
    pump.classList.toggle("rotate", (data.action || []).some((v) => v > 0));
  }

  const riskEl = $("risk");
  if (riskEl) {
    riskEl.classList.remove("risk-low", "risk-medium", "risk-high");
    const risk = String(data.risk || "").toLowerCase();
    if (risk === "low" || risk === "medium" || risk === "high") {
      riskEl.classList.add(`risk-${risk}`);
    }
  }

  $("alert").style.display = (data.overflow ?? 0) > 0 ? "block" : "none";
}

function updateReplay() {
  const slider = $("replay");
  slider.max = Math.max(0, state.history.length - 1);
  slider.value = Math.max(0, state.history.length - 1);
  $("replayLabel").textContent = `${state.history.length}/${state.history.length}`;
  if (state.history.length > 0) {
    renderCurrent(state.history[state.history.length - 1]);
  }
}

function renderAtIndex(idx) {
  const item = state.history[idx];
  if (!item) return;
  renderCurrent(item);
  $("replayLabel").textContent = `${idx + 1}/${state.history.length}`;
}

function renderCharts() {
  if (!state.timelineChart) {
    state.timelineChart = echarts.init($("timeline"));
  }
  if (!state.compareChart) {
    state.compareChart = echarts.init($("compare"));
  }

  const x = state.history.map((_, i) => i + 1);
  const rain = state.history.map((d) => d.rain);
  const reward = state.history.map((d) => d.reward);
  const overflow = state.history.map((d) => d.overflow);

  state.timelineChart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["降雨", "奖励", "溢流"], top: 8 },
    xAxis: { type: "category", data: x },
    yAxis: [{ type: "value", splitLine: { lineStyle: { color: "rgba(16,34,58,0.12)" } } }],
    series: [
      {
        name: "降雨",
        type: "line",
        data: rain,
        smooth: true,
        lineStyle: { width: 2, color: "#2bd0ff" },
        itemStyle: { color: "#2bd0ff" },
      },
      {
        name: "奖励",
        type: "line",
        data: reward,
        smooth: true,
        lineStyle: { width: 2, color: "#1eb980" },
        itemStyle: { color: "#1eb980" },
      },
      {
        name: "溢流",
        type: "line",
        data: overflow,
        smooth: true,
        lineStyle: { width: 2, color: "#ef5350" },
        itemStyle: { color: "#ef5350" },
      },
    ],
  });
}

async function loadCompare() {
  const res = await fetch("/api/sim/compare?steps=60&seed=42");
  const data = await res.json();
  if (!state.compareChart) {
    state.compareChart = echarts.init($("compare"));
  }
  state.compareChart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["规则", "RL"], top: 8 },
    xAxis: { type: "category", data: ["总溢流", "总能耗", "平均奖励"] },
    yAxis: { type: "value", splitLine: { lineStyle: { color: "rgba(16,34,58,0.12)" } } },
    series: [
      {
        name: "规则",
        type: "bar",
        itemStyle: { color: "#f6a31a" },
        data: [
          data.rule_metrics.total_overflow,
          data.rule_metrics.total_energy,
          data.rule_metrics.avg_reward,
        ],
      },
      {
        name: "RL",
        type: "bar",
        itemStyle: { color: "#15a7d9" },
        data: [
          data.rl_metrics.total_overflow,
          data.rl_metrics.total_energy,
          data.rl_metrics.avg_reward,
        ],
      },
    ],
  });
}

async function stepOnce() {
  if (!state.sessionId) {
    await initSession();
  }
  const res = await fetch("/api/sim/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      manual_action: state.mode === "manual" ? state.manualAction : undefined,
    }),
  });
  const data = await res.json();
  state.history.push(data);
  renderCurrent(data);
  updateReplay();
  renderCharts();
}

function startLoop() {
  if (state.running) return;
  state.running = true;
  $("toggleBtn").textContent = "暂停";
  state.interval = setInterval(() => {
    stepOnce();
  }, 1000);
}

function stopLoop() {
  state.running = false;
  $("toggleBtn").textContent = "继续";
  if (state.interval) clearInterval(state.interval);
}

async function resetSim() {
  stopLoop();
  state.history = [];
  if (!state.sessionId) {
    await initSession();
    return;
  }
  const res = await fetch("/api/sim/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId }),
  });
  const data = await res.json();
  state.history = [data];
  renderCurrent(data);
  updateReplay();
  renderCharts();
}

document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    state.manualAction = [0, 0, 0];
    syncManualButtons();
    setActiveMode(btn.dataset.mode);
    await initSession();
    await resetSim();
  });
});

$("initBtn").addEventListener("click", async () => {
  await initSession();
  await stepOnce();
  startLoop();
});

$("toggleBtn").addEventListener("click", () => {
  if (state.running) stopLoop();
  else startLoop();
});

$("resetBtn").addEventListener("click", resetSim);

$("replay").addEventListener("input", (e) => {
  const idx = Number(e.target.value || 0);
  renderAtIndex(idx);
});

document.querySelectorAll(".manual-btn[data-pump]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (state.mode !== "manual") return;
    const idx = Number(btn.dataset.pump || -1);
    if (idx < 0 || idx > 2) return;
    state.manualAction[idx] = state.manualAction[idx] ? 0 : 1;
    syncManualButtons();
    if (!state.running) await stepOnce();
  });
});

$("manualAllOn")?.addEventListener("click", async () => {
  if (state.mode !== "manual") return;
  state.manualAction = [1, 1, 1];
  syncManualButtons();
  if (!state.running) await stepOnce();
});

$("manualAllOff")?.addEventListener("click", async () => {
  if (state.mode !== "manual") return;
  state.manualAction = [0, 0, 0];
  syncManualButtons();
  if (!state.running) await stepOnce();
});

window.addEventListener("resize", () => {
  state.timelineChart?.resize();
  state.compareChart?.resize();
});

setActiveMode("rl");
syncManualButtons();
setPumpState([0, 0, 0]);
loadCompare();
initSession().then(() => resetSim());
