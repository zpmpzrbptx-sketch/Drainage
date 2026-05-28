const state = {
  timer: null,
  charts: {},
};

function byId(id) {
  return document.getElementById(id);
}

function fmt(v, d = 2) {
  const n = Number(v || 0);
  return Number.isFinite(n) ? n.toFixed(d) : "0.00";
}

function avg(arr = []) {
  if (!arr.length) return 0;
  return arr.reduce((s, v) => s + Number(v || 0), 0) / arr.length;
}

function max(arr = []) {
  if (!arr.length) return 0;
  return Math.max(...arr.map((v) => Number(v || 0)));
}

function normalizePayload(payload = {}) {
  const fallbackSeries = {
    rain: Array.isArray(payload.rain) ? payload.rain : [],
    water: Array.isArray(payload.water) ? payload.water : [[], [], []],
    energy: Array.isArray(payload.energy) ? payload.energy : [],
    overflow: [],
    reward: [],
    flow: [[], [], []],
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
    },
    compare: payload.compare || {
      categories: ["总溢流", "总能耗", "平均奖励"],
      rule: [0, 0, 0],
      rl: [0, 0, 0],
    },
    kpi: payload.kpi || {
      rain_now: avg(fallbackSeries.rain),
      avg_water_now: avg([
        ...fallbackSeries.water[0],
        ...fallbackSeries.water[1],
        ...fallbackSeries.water[2],
      ]),
      peak_water: max([
        ...fallbackSeries.water[0],
        ...fallbackSeries.water[1],
        ...fallbackSeries.water[2],
      ]),
      total_energy: avg(fallbackSeries.energy),
      total_overflow: 0,
      risk_score: 0,
      system_score: 100,
      pump_on_ratio: [0, 0, 0],
    },
    events: Array.isArray(payload.events) ? payload.events : [],
    meta: payload.meta || { step: 0, mode: "rule" },
  };
}

function ensureCharts() {
  if (Object.keys(state.charts).length) return;
  state.charts.rain = echarts.init(byId("rain"));
  state.charts.water = echarts.init(byId("water"));
  state.charts.energy = echarts.init(byId("energy"));
  state.charts.compare = echarts.init(byId("compare"));
  state.charts.riskGauge = echarts.init(byId("riskGauge"));
  state.charts.heatmap = echarts.init(byId("heatmap"));
  state.charts.pumpMix = echarts.init(byId("pumpMix"));
  state.charts.radar = echarts.init(byId("radar"));
}

function setKpis(payload) {
  const { kpi = {}, meta = {} } = payload;
  byId("kpiRain").textContent = `${fmt(kpi.rain_now)} mm/h`;
  byId("kpiWater").textContent = `${fmt(kpi.avg_water_now)} m`;
  byId("kpiPeak").textContent = `${fmt(kpi.peak_water)} m`;
  byId("kpiEnergy").textContent = `${fmt(kpi.total_energy)} kWh`;
  byId("kpiOverflow").textContent = `${fmt(kpi.total_overflow)} m³`;
  byId("kpiScore").textContent = fmt(kpi.system_score, 0);
  byId("updateTime").textContent = `最近更新：${new Date().toLocaleTimeString("zh-CN", { hour12: false })} | Step ${meta.step ?? 0}`;
}

function renderRain(series, timeline) {
  state.charts.rain.setOption({
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: timeline, boundaryGap: false },
    yAxis: { type: "value", name: "mm/h" },
    series: [
      {
        name: "降雨",
        type: "line",
        smooth: true,
        data: series.rain,
        lineStyle: { width: 2, color: "#25d6ff" },
        itemStyle: { color: "#25d6ff" },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: "rgba(37,214,255,0.46)" },
            { offset: 1, color: "rgba(37,214,255,0.06)" },
          ]),
        },
      },
    ],
  });
}

function renderWater(series, timeline) {
  const water = Array.isArray(series.water) ? series.water : [[], [], []];
  const meanWater = timeline.map((_, i) => {
    const w1 = Number(water[0]?.[i] || 0);
    const w2 = Number(water[1]?.[i] || 0);
    const w3 = Number(water[2]?.[i] || 0);
    return (w1 + w2 + w3) / 3;
  });

  state.charts.water.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["节点1", "节点2", "节点3", "平均水位"] },
    xAxis: { type: "category", data: timeline },
    yAxis: { type: "value", name: "m" },
    series: [
      { name: "节点1", type: "line", smooth: true, data: water[0], lineStyle: { color: "#6ee7ff" } },
      { name: "节点2", type: "line", smooth: true, data: water[1], lineStyle: { color: "#7effc5" } },
      { name: "节点3", type: "line", smooth: true, data: water[2], lineStyle: { color: "#ffd166" } },
      { name: "平均水位", type: "line", smooth: true, data: meanWater, lineStyle: { type: "dashed", color: "#ff7b7b" } },
      {
        name: "安全阈值",
        type: "line",
        data: Array(timeline.length).fill(6),
        symbol: "none",
        lineStyle: { color: "#ef5350", opacity: 0.5 },
        tooltip: { show: false },
      },
    ],
  });
}

function renderEnergy(series, timeline) {
  const energy = series.energy || [];
  const rolling = energy.map((_, i) => avg(energy.slice(Math.max(0, i - 2), i + 1)));

  state.charts.energy.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["能耗", "3步均值"] },
    xAxis: { type: "category", data: timeline },
    yAxis: { type: "value", name: "kWh" },
    series: [
      { name: "能耗", type: "bar", data: energy, itemStyle: { color: "rgba(114, 184, 255, 0.85)" } },
      { name: "3步均值", type: "line", smooth: true, data: rolling, lineStyle: { color: "#f6a623" } },
    ],
  });
}

function renderCompare(compare) {
  state.charts.compare.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["规则", "RL"] },
    xAxis: { type: "category", data: compare.categories || ["总溢流", "总能耗", "平均奖励"] },
    yAxis: { type: "value" },
    series: [
      { name: "规则", type: "bar", data: compare.rule || [], itemStyle: { color: "#f6a623" } },
      { name: "RL", type: "bar", data: compare.rl || [], itemStyle: { color: "#25d6ff" } },
    ],
  });
}

function renderRiskGauge(kpi) {
  const riskScore = Number(kpi.risk_score || 0);
  state.charts.riskGauge.setOption({
    series: [
      {
        type: "gauge",
        min: 0,
        max: 100,
        axisLine: {
          lineStyle: {
            width: 16,
            color: [[0.45, "#18b97d"], [0.75, "#f6a623"], [1, "#ef5350"]],
          },
        },
        progress: { show: true, width: 16 },
        detail: { valueAnimation: true, formatter: "{value}", color: "#e5f4ff", fontSize: 28 },
        data: [{ value: Number(fmt(riskScore, 0)), name: "风险指数" }],
        title: { color: "#9dc1d9", fontSize: 12 },
      },
    ],
  });
}

function renderHeatmap(series) {
  const rain = Array.isArray(series.rain) ? series.rain : [];
  const water = Array.isArray(series.water) ? series.water : [[], [], []];
  const xLabels = Array.from({ length: rain.length }, (_, i) => `T${i + 1}`);
  const yLabels = ["节点1", "节点2", "节点3"];
  const heat = [];
  yLabels.forEach((_, row) => {
    xLabels.forEach((__, col) => {
      heat.push([col, row, Number(water[row]?.[col] || 0)]);
    });
  });

  state.charts.heatmap.setOption({
    tooltip: {
      position: "top",
      formatter: (p) => `${yLabels[p.value[1]]} ${xLabels[p.value[0]]}<br/>水位: ${fmt(p.value[2])} m`,
    },
    grid: { top: 34, left: 56, right: 10, bottom: 40 },
    xAxis: { type: "category", data: xLabels, splitArea: { show: true } },
    yAxis: { type: "category", data: yLabels, splitArea: { show: true } },
    visualMap: {
      min: 0,
      max: Math.max(10, max([...(water[0] || []), ...(water[1] || []), ...(water[2] || [])])),
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      inRange: { color: ["#0b4064", "#2fb5ff", "#ffe082", "#ef5350"] },
      textStyle: { color: "#9fc2da" },
    },
    series: [{ type: "heatmap", data: heat }],
  });
}

function renderPumpMix(kpi) {
  const onRatio = kpi.pump_on_ratio || [0, 0, 0];
  const offRatio = onRatio.map((v) => Math.max(0, 100 - Number(v || 0)));
  state.charts.pumpMix.setOption({
    tooltip: { trigger: "item" },
    legend: { top: "bottom", textStyle: { color: "#a7c4d9" } },
    series: [
      {
        type: "pie",
        radius: ["45%", "70%"],
        label: { color: "#d9ecff" },
        data: [
          { name: "泵站1开启占比", value: onRatio[0], itemStyle: { color: "#25d6ff" } },
          { name: "泵站2开启占比", value: onRatio[1], itemStyle: { color: "#7effc5" } },
          { name: "泵站3开启占比", value: onRatio[2], itemStyle: { color: "#ffd166" } },
          { name: "关闭余量", value: avg(offRatio), itemStyle: { color: "#425e78" } },
        ],
      },
    ],
  });
}

function renderRadar(payload) {
  const { series = {}, kpi = {}, compare = {} } = payload;
  const rainStability = Math.max(0, 100 - (max(series.rain || []) - avg(series.rain || [])) * 8);
  const waterSafety = Math.max(0, 100 - Math.max(0, Number(kpi.avg_water_now || 0) - 4) * 20);
  const energyEfficiency = Math.max(0, 100 - Number(kpi.total_energy || 0) * 0.8);
  const overflowControl = Math.max(0, 100 - Number(kpi.total_overflow || 0) * 10);
  const strategyScore = Math.max(0, 100 - Math.max(0, Number(compare.rule?.[0] || 0) - Number(compare.rl?.[0] || 0)) * 8);

  state.charts.radar.setOption({
    radar: {
      indicator: [
        { name: "降雨稳定", max: 100 },
        { name: "水位安全", max: 100 },
        { name: "能耗效率", max: 100 },
        { name: "溢流控制", max: 100 },
        { name: "策略优势", max: 100 },
      ],
      splitLine: { lineStyle: { color: "rgba(180, 214, 239, 0.25)" } },
      axisLine: { lineStyle: { color: "rgba(180, 214, 239, 0.35)" } },
    },
    series: [
      {
        type: "radar",
        data: [
          {
            value: [rainStability, waterSafety, energyEfficiency, overflowControl, strategyScore],
            areaStyle: { color: "rgba(37,214,255,0.22)" },
            lineStyle: { color: "#25d6ff" },
          },
        ],
      },
    ],
  });
}

function renderEvents(events = []) {
  const html = events
    .map((e) => `<div class="event-item ${e.level || "safe"}"><div class="event-time">${e.time || "--"}</div><div class="event-text">${e.message || ""}</div></div>`)
    .join("");
  byId("eventFeed").innerHTML = html;
}

async function loadData() {
  try {
    const res = await fetch('/api/data?steps=20');
    const raw = await res.json();
    const payload = normalizePayload(raw);

    ensureCharts();

    const series = payload.series || {};
    const timeline = Array.from({ length: (series.rain || []).length }, (_, i) => i + 1);

    setKpis(payload);
    renderRain(series, timeline);
    renderWater(series, timeline);
    renderEnergy(series, timeline);
    renderCompare(payload.compare || {});
    renderRiskGauge(payload.kpi || {});
    renderHeatmap(series);
    renderPumpMix(payload.kpi || {});
    renderRadar(payload);
    renderEvents(payload.events || []);
  } catch (err) {
    byId("eventFeed").innerHTML = '<div class="event-item danger"><div class="event-time">ERROR</div><div class="event-text">数据加载失败，请检查后端服务。</div></div>';
  }
}

function start() {
  byId("refreshBtn")?.addEventListener("click", loadData);
  loadData();
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(loadData, 3000);
}

window.addEventListener("resize", () => {
  Object.values(state.charts).forEach((c) => c.resize());
});

start();
