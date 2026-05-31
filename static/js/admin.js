const state = {
  user: null,
  page: 1,
  pageSize: 20,
  total: 0,
  records: [],
  users: [],
  charts: {},
};

function byId(id) {
  return document.getElementById(id);
}

function hasEl(id) {
  return Boolean(byId(id));
}

function fmt(v, d = 2) {
  const n = Number(v || 0);
  return Number.isFinite(n) ? n.toFixed(d) : '0.00';
}

function esc(v) {
  return String(v ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function showToast(text) {
  const el = byId('toast');
  el.textContent = text;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 2200);
}

function isValidUsername(username) {
  return /^[A-Za-z0-9_]+$/.test(username);
}

function calcPasswordStrength(password) {
  const pwd = String(password || '');
  if (!pwd) return { label: '-', level: 'none' };
  let score = 0;
  if (pwd.length >= 6) score += 1;
  if (pwd.length >= 10) score += 1;
  if (/[a-z]/.test(pwd) && /[A-Z]/.test(pwd)) score += 1;
  if (/\d/.test(pwd)) score += 1;
  if (/[^A-Za-z0-9]/.test(pwd)) score += 1;
  if (score <= 2) return { label: '弱', level: 'weak' };
  if (score <= 4) return { label: '中', level: 'medium' };
  return { label: '强', level: 'strong' };
}

function updatePasswordStrengthTip() {
  if (!hasEl('passwordStrengthTip') || !hasEl('loginPassword')) return;
  const { label } = calcPasswordStrength(byId('loginPassword').value || '');
  byId('passwordStrengthTip').textContent = `密码强度：${label}`;
}

function toLocalInputValue(s) {
  if (!s) return '';
  const date = new Date(String(s).replace(' ', 'T'));
  if (Number.isNaN(date.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function toApiDateTime(localVal) {
  if (!localVal) return '';
  return `${localVal.replace('T', ' ')}:00`;
}

async function api(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = body.message || body.error || '请求失败';
    throw new Error(msg);
  }
  return body;
}

function setLoggedIn(loggedIn) {
  if (hasEl('loginSection')) byId('loginSection').classList.toggle('hidden', loggedIn);
  if (hasEl('dashboardSection')) byId('dashboardSection').classList.toggle('hidden', !loggedIn);
  if (hasEl('logoutBtn')) byId('logoutBtn').classList.toggle('hidden', !loggedIn);
}

function updateWhoAmI() {
  if (!hasEl('whoami')) return;
  if (!state.user) {
    byId('whoami').textContent = '未登录';
    return;
  }
  byId('whoami').textContent = `当前用户: ${state.user.username} (${state.user.role})`;
}

function buildQuery() {
  const p = new URLSearchParams();
  p.set('page', String(state.page));
  p.set('page_size', String(state.pageSize));

  const map = [
    ['risk_level', 'fRiskLevel'],
    ['keyword', 'fKeyword'],
    ['min_rain', 'fMinRain'],
    ['max_rain', 'fMaxRain'],
    ['min_inflow', 'fMinInflow'],
    ['max_inflow', 'fMaxInflow'],
    ['min_overflow', 'fMinOverflow'],
    ['max_overflow', 'fMaxOverflow'],
    ['min_water', 'fMinWater'],
    ['max_water', 'fMaxWater'],
  ];
  map.forEach(([k, id]) => {
    const v = byId(id).value.trim();
    if (v) p.set(k, v);
  });

  const start = toApiDateTime(byId('fStartTime').value);
  const end = toApiDateTime(byId('fEndTime').value);
  if (start) p.set('start_time', start);
  if (end) p.set('end_time', end);
  p.set('sort_by', 'record_time');
  p.set('sort_order', 'desc');
  return p;
}

function fillRecordTable(items = []) {
  const tbody = byId('recordTableBody');
  tbody.innerHTML = items.map((r) => {
    const avgWater = (Number(r.water_node1 || 0) + Number(r.water_node2 || 0) + Number(r.water_node3 || 0)) / 3;
    const rid = Number(r.id || 0);
    return `
      <tr>
        <td>${rid}</td>
        <td>${esc(r.record_time || '')}</td>
        <td>${fmt(r.rain)}</td>
        <td>${fmt(r.inflow)}</td>
        <td>${fmt(avgWater)}</td>
        <td>${fmt(r.overflow)}</td>
        <td><span class="tag ${esc(r.risk_level || 'low')}">${esc(r.risk_level || 'low')}</span></td>
        <td>
          <button class="admin-btn ghost" onclick="openEditRecord(${rid})">编辑</button>
          ${state.user?.role === 'admin' ? `<button class="admin-btn ghost" onclick="deleteRecord(${rid})">删除</button>` : ''}
        </td>
      </tr>
    `;
  }).join('');
}

function updatePager() {
  const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
  byId('pageInfo').textContent = `${state.page} / ${totalPages}`;
  byId('recordCount').textContent = `共 ${state.total} 条`;
}

async function loadRecords() {
  const p = buildQuery();
  const data = await api(`/api/admin/records?${p.toString()}`);
  state.records = data.items || [];
  state.total = Number(data.total || 0);
  fillRecordTable(state.records);
  updatePager();
}

function ensureCharts() {
  if (Object.keys(state.charts).length) return;
  if (typeof window.echarts === 'undefined') {
    throw new Error('图表库加载失败，请检查网络或刷新页面。');
  }
  state.charts.summary = echarts.init(byId('summaryChart'));
  state.charts.risk = echarts.init(byId('riskChart'));
  state.charts.mode = echarts.init(byId('modeChart'));
}

function setKpis(kpi = {}) {
  if (hasEl('kpiTotal')) byId('kpiTotal').textContent = String(kpi.total || 0);
  if (hasEl('kpiRain')) byId('kpiRain').textContent = fmt(kpi.avg_rain);
  if (hasEl('kpiWater')) byId('kpiWater').textContent = fmt(kpi.avg_water);
  if (hasEl('kpiPeak')) byId('kpiPeak').textContent = fmt(kpi.peak_water);
  if (hasEl('kpiInflow')) {
    byId('kpiInflow').textContent = fmt(kpi.avg_inflow);
  } else if (hasEl('kpiEnergy')) {
    byId('kpiEnergy').textContent = fmt(kpi.total_energy);
  }
  if (hasEl('kpiOverflow')) byId('kpiOverflow').textContent = fmt(kpi.total_overflow);
}

function renderSummaryCharts(data = {}) {
  ensureCharts();
  const rows = Array.isArray(data.series) ? data.series : [];
  const x = rows.map((r) => r.bucket || '');

  state.charts.summary.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['平均降雨', '平均水位', '总溢流'] },
    xAxis: { type: 'category', data: x },
    yAxis: [{ type: 'value', name: 'rain/water' }, { type: 'value', name: 'overflow' }],
    series: [
      { name: '平均降雨', type: 'line', smooth: true, data: rows.map((r) => Number(r.avg_rain || 0)) },
      { name: '平均水位', type: 'line', smooth: true, data: rows.map((r) => Number(r.avg_water || 0)) },
      { name: '总溢流', type: 'bar', yAxisIndex: 1, data: rows.map((r) => Number(r.total_overflow || 0)) },
    ],
  });

  const riskRows = Array.isArray(data.risk_distribution) ? data.risk_distribution : [];
  state.charts.risk.setOption({
    tooltip: { trigger: 'item' },
    series: [{
      type: 'pie',
      radius: ['45%', '70%'],
      data: riskRows.map((r) => ({ name: r.risk_level || 'unknown', value: Number(r.count || 0) })),
    }],
  });

  const modeRows = Array.isArray(data.mode_distribution) ? data.mode_distribution : [];
  state.charts.mode.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: modeRows.map((r) => r.range || '') },
    yAxis: { type: 'value' },
    series: [{ type: 'bar', data: modeRows.map((r) => Number(r.count || 0)) }],
  });
}

async function loadSummary() {
  const p = buildQuery();
  p.delete('page');
  p.delete('page_size');
  const data = await api(`/api/admin/summary?${p.toString()}`);
  setKpis(data.kpi || {});
  renderSummaryCharts(data);
}

function clearRecordModal() {
  byId('recordId').value = '';
  ['mScenario', 'mRecordTime', 'mMode', 'mRain', 'mInflow', 'mOverflow', 'mEnergy', 'mReward', 'mW1', 'mW2', 'mW3', 'mF1', 'mF2', 'mF3', 'mS1', 'mS2', 'mS3'].forEach((id) => byId(id).value = '');
  byId('mRisk').value = '';
}

function openRecordModal(edit = false) {
  byId('recordModalTitle').textContent = edit ? '编辑记录' : '新增记录';
  byId('recordModal').classList.remove('hidden');
}

function closeRecordModal() {
  byId('recordModal').classList.add('hidden');
}

async function openEditRecord(id) {
  try {
    const data = await api(`/api/admin/records/${id}`);
    const r = data.item || {};
    byId('recordId').value = String(r.id || '');
    byId('mScenario').value = r.scenario || '';
    byId('mRecordTime').value = toLocalInputValue(r.record_time);
    byId('mMode').value = r.mode || '';
    byId('mRain').value = r.rain ?? '';
    byId('mInflow').value = r.inflow ?? '';
    byId('mOverflow').value = r.overflow ?? '';
    byId('mEnergy').value = r.energy ?? '';
    byId('mReward').value = r.reward ?? '';
    byId('mW1').value = r.water_node1 ?? '';
    byId('mW2').value = r.water_node2 ?? '';
    byId('mW3').value = r.water_node3 ?? '';
    byId('mF1').value = r.flow_node1 ?? '';
    byId('mF2').value = r.flow_node2 ?? '';
    byId('mF3').value = r.flow_node3 ?? '';
    byId('mS1').value = r.storage_node1 ?? '';
    byId('mS2').value = r.storage_node2 ?? '';
    byId('mS3').value = r.storage_node3 ?? '';
    byId('mRisk').value = r.risk_level || '';
    openRecordModal(true);
  } catch (e) {
    showToast(e.message || '加载失败');
  }
}

function numOrNull(v) {
  const t = String(v ?? '').trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

async function saveRecord() {
  const id = byId('recordId').value.trim();
  const payload = {
    scenario: byId('mScenario').value.trim() || 'default',
    record_time: toApiDateTime(byId('mRecordTime').value),
    mode: byId('mMode').value.trim() || 'rule',
    rain: numOrNull(byId('mRain').value),
    inflow: numOrNull(byId('mInflow').value),
    overflow: numOrNull(byId('mOverflow').value),
    energy: numOrNull(byId('mEnergy').value),
    reward: numOrNull(byId('mReward').value),
    water_node1: numOrNull(byId('mW1').value),
    water_node2: numOrNull(byId('mW2').value),
    water_node3: numOrNull(byId('mW3').value),
    flow_node1: numOrNull(byId('mF1').value),
    flow_node2: numOrNull(byId('mF2').value),
    flow_node3: numOrNull(byId('mF3').value),
    storage_node1: numOrNull(byId('mS1').value),
    storage_node2: numOrNull(byId('mS2').value),
    storage_node3: numOrNull(byId('mS3').value),
    risk_level: byId('mRisk').value.trim(),
  };

  Object.keys(payload).forEach((k) => {
    if (payload[k] === null || payload[k] === '') delete payload[k];
  });

  try {
    if (!id) {
      await api('/api/admin/records', { method: 'POST', body: JSON.stringify(payload) });
      showToast('新增成功');
    } else {
      await api(`/api/admin/records/${id}`, { method: 'PUT', body: JSON.stringify(payload) });
      showToast('更新成功');
    }
    closeRecordModal();
    await Promise.all([loadRecords(), loadSummary()]);
  } catch (e) {
    showToast(e.message || '保存失败');
  }
}

async function deleteRecord(id) {
  if (state.user?.role !== 'admin') {
    showToast('仅管理员可删除');
    return;
  }
  if (!confirm(`确认删除记录 #${id} ?`)) return;
  try {
    await api(`/api/admin/records/${id}`, { method: 'DELETE' });
    showToast('删除成功');
    await Promise.all([loadRecords(), loadSummary()]);
  } catch (e) {
    showToast(e.message || '删除失败');
  }
}

async function loadUsers() {
  if (state.user?.role !== 'admin') {
    byId('userSection').classList.add('hidden');
    return;
  }
  byId('userSection').classList.remove('hidden');
  const data = await api('/api/admin/users');
  state.users = data.items || [];
  const tbody = byId('userTableBody');
  tbody.innerHTML = state.users.map((u) => `
    <tr>
      <td>${Number(u.id || 0)}</td>
      <td>${esc(u.username)}</td>
      <td>${esc(u.role)}</td>
      <td>${Number(u.is_active) ? '启用' : '禁用'}</td>
      <td>${esc(u.created_at || '')}</td>
      <td>
        <button class="admin-btn ghost" onclick="openEditUser(${Number(u.id || 0)})">编辑</button>
        <button class="admin-btn ghost" onclick="deleteUser(${Number(u.id || 0)})">删除</button>
      </td>
    </tr>
  `).join('');
}

function clearUserModal() {
  byId('userId').value = '';
  byId('uName').value = '';
  byId('uRole').value = 'user';
  byId('uPassword').value = '';
  byId('uActive').value = '1';
}

function openUserModal(edit = false) {
  byId('userModalTitle').textContent = edit ? '编辑用户' : '新增用户';
  byId('userModal').classList.remove('hidden');
}

function closeUserModal() {
  byId('userModal').classList.add('hidden');
}

function openEditUser(id) {
  const u = state.users.find((x) => Number(x.id) === Number(id));
  if (!u) return;
  byId('userId').value = String(u.id || '');
  byId('uName').value = u.username || '';
  byId('uRole').value = u.role || 'user';
  byId('uPassword').value = '';
  byId('uActive').value = Number(u.is_active) ? '1' : '0';
  openUserModal(true);
}

async function saveUser() {
  const id = byId('userId').value.trim();
  const payload = {
    username: byId('uName').value.trim(),
    role: byId('uRole').value,
    is_active: byId('uActive').value === '1',
    password: byId('uPassword').value,
  };

  try {
    if (!id) {
      if (!payload.username || !payload.password) {
        showToast('新增用户需填写用户名和密码');
        return;
      }
      await api('/api/admin/users', { method: 'POST', body: JSON.stringify(payload) });
      showToast('用户创建成功');
    } else {
      const up = { role: payload.role, is_active: payload.is_active };
      if (payload.password) up.password = payload.password;
      await api(`/api/admin/users/${id}`, { method: 'PUT', body: JSON.stringify(up) });
      showToast('用户更新成功');
    }
    closeUserModal();
    await loadUsers();
  } catch (e) {
    showToast(e.message || '用户保存失败');
  }
}

async function deleteUser(id) {
  if (!confirm(`确认删除用户 #${id} ?`)) return;
  try {
    await api(`/api/admin/users/${id}`, { method: 'DELETE' });
    showToast('删除成功');
    await loadUsers();
  } catch (e) {
    showToast(e.message || '删除失败');
  }
}

async function doLogin() {
  const username = byId('loginUsername').value.trim();
  const password = byId('loginPassword').value.trim();
  if (!username || !password) {
    showToast('请输入用户名和密码');
    return;
  }
  try {
    const data = await api('/api/admin/login', { method: 'POST', body: JSON.stringify({ username, password }) });
    state.user = data.user || null;
    if (window.location.pathname === '/admin/login') {
      window.location.href = '/';
      return;
    }
    updateWhoAmI();
    setLoggedIn(true);
    await bootstrapDashboard();
    showToast('登录成功');
  } catch (e) {
    showToast(e.message || '登录失败');
  }
}

async function doRegister() {
  const username = byId('loginUsername')?.value.trim() || '';
  const password = byId('loginPassword')?.value.trim() || '';
  const confirmPassword = byId('confirmPassword')?.value.trim() || '';
  if (!username || !password || !confirmPassword) {
    showToast('请输入用户名、密码和确认密码');
    return;
  }
  if (!isValidUsername(username)) {
    showToast('用户名仅允许字母/数字/下划线');
    return;
  }
  if (password !== confirmPassword) {
    showToast('两次密码输入不一致');
    return;
  }
  if (password.length < 6) {
    showToast('请输入用户名和密码');
    return;
  }
  try {
    const data = await api('/api/admin/register', { method: 'POST', body: JSON.stringify({ username, password }) });
    showToast(data.message || '注册成功，请登录');
    if (window.location.pathname === '/admin/register') {
      setTimeout(() => {
        window.location.href = '/admin/login';
      }, 500);
    }
  } catch (e) {
    showToast(e.message || '注册失败');
  }
}

async function doInit() {
  const username = byId('loginUsername').value.trim() || 'admin';
  const password = byId('loginPassword').value.trim() || 'admin123456';
  try {
    const data = await api('/api/admin/init', { method: 'POST', body: JSON.stringify({ username, password }) });
    showToast(data.message || '初始化完成');
  } catch (e) {
    showToast(e.message || '初始化失败');
  }
}

async function doLogout() {
  await api('/api/admin/logout', { method: 'POST' });
  state.user = null;
  updateWhoAmI();
  setLoggedIn(false);
}

async function checkLogin() {
  try {
    const data = await api('/api/admin/me');
    if (data.logged_in) {
      state.user = data.user;
      if (window.location.pathname === '/admin/login') {
        window.location.href = '/';
        return;
      }
      updateWhoAmI();
      setLoggedIn(true);
      await bootstrapDashboard();
    } else {
      state.user = null;
      updateWhoAmI();
      setLoggedIn(false);
    }
  } catch (_) {
    state.user = null;
    setLoggedIn(false);
  }
}

async function bootstrapDashboard() {
  await loadRecords();
  try {
    await loadSummary();
  } catch (e) {
    showToast(e.message || '图表加载失败，已显示基础数据');
  }
  await loadUsers();
}

function resetFilters() {
  ['fRiskLevel', 'fStartTime', 'fEndTime', 'fKeyword', 'fMinRain', 'fMaxRain', 'fMinInflow', 'fMaxInflow', 'fMinOverflow', 'fMaxOverflow', 'fMinWater', 'fMaxWater'].forEach((id) => {
    byId(id).value = '';
  });
  state.page = 1;
}

function bindEvents() {
  if (hasEl('loginBtn')) byId('loginBtn').addEventListener('click', doLogin);
  if (hasEl('registerBtn')) byId('registerBtn').addEventListener('click', doRegister);
  if (hasEl('initBtn')) byId('initBtn').addEventListener('click', doInit);
  if (hasEl('logoutBtn')) byId('logoutBtn').addEventListener('click', doLogout);

  if (hasEl('queryBtn')) byId('queryBtn').addEventListener('click', async () => {
    state.page = 1;
    try {
      await Promise.all([loadRecords(), loadSummary()]);
    } catch (e) {
      showToast(e.message || '查询成功，但统计图加载失败');
    }
  });

  if (hasEl('resetFilterBtn')) byId('resetFilterBtn').addEventListener('click', async () => {
    resetFilters();
    try {
      await Promise.all([loadRecords(), loadSummary()]);
    } catch (e) {
      showToast(e.message || '重置成功，但统计图加载失败');
    }
  });

  if (hasEl('prevPageBtn')) byId('prevPageBtn').addEventListener('click', async () => {
    if (state.page <= 1) return;
    state.page -= 1;
    await loadRecords();
  });

  if (hasEl('nextPageBtn')) byId('nextPageBtn').addEventListener('click', async () => {
    const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
    if (state.page >= totalPages) return;
    state.page += 1;
    await loadRecords();
  });

  if (hasEl('newRecordBtn')) byId('newRecordBtn').addEventListener('click', () => {
    clearRecordModal();
    openRecordModal(false);
  });

  if (hasEl('cancelRecordBtn')) byId('cancelRecordBtn').addEventListener('click', closeRecordModal);
  if (hasEl('saveRecordBtn')) byId('saveRecordBtn').addEventListener('click', saveRecord);

  if (hasEl('newUserBtn')) byId('newUserBtn').addEventListener('click', () => {
    clearUserModal();
    openUserModal(false);
  });

  if (hasEl('cancelUserBtn')) byId('cancelUserBtn').addEventListener('click', closeUserModal);
  if (hasEl('saveUserBtn')) byId('saveUserBtn').addEventListener('click', saveUser);
  if (hasEl('loginPassword')) byId('loginPassword').addEventListener('input', updatePasswordStrengthTip);
  updatePasswordStrengthTip();

  window.addEventListener('resize', () => {
    Object.values(state.charts).forEach((c) => c.resize());
  });
}

window.openEditRecord = openEditRecord;
window.deleteRecord = deleteRecord;
window.openEditUser = openEditUser;
window.deleteUser = deleteUser;

setLoggedIn(Boolean(hasEl('dashboardSection')));
updateWhoAmI();
bindEvents();
checkLogin();
