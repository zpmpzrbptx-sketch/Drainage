import re
from pathlib import Path
from typing import Any, Dict, Optional, cast

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class DrainageEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    排水系统“厂-站-网”联合调度环境。

    支持两种数据源：
    1) random：当前默认，使用内置简化动力学
    2) swmm_csv：从 SWMM 导出的时序 CSV 读取外源数据

    状态向量（长度10）：
    [rain, water_1, water_2, water_3, flow_1, flow_2, flow_3, storage_1, storage_2, storage_3]

    动作向量（长度3，离散0/1）：
    [pump_1, pump_2, pump_3]，0=关泵，1=开泵
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        swmm_csv_path: Optional[str] = None,
        max_steps: int = 200,
        severe_overflow_threshold: float = 12.0,
        random_start: bool = False,
        high_water_threshold: float = 5.0,
        medium_water_threshold: float = 3.0,
    ):
        super().__init__()

        self.state_dim = 10
        self.action_dim = 3

        obs_low = np.array([0.0] * self.state_dim, dtype=np.float32)
        obs_high = np.array(
            [50.0, 15.0, 15.0, 15.0, 30.0, 30.0, 30.0, 120.0, 120.0, 120.0],
            dtype=np.float32,
        )
        # 观测上下界：用于训练时归一化/裁剪，避免状态出现异常值
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        # 三个泵站分别二值开关，共 2^3=8 种组合动作
        self.action_space = spaces.MultiDiscrete([2, 2, 2])

        self.max_steps = int(max_steps)
        self.severe_overflow_threshold = float(severe_overflow_threshold)
        self.random_start = bool(random_start)
        self.high_water_threshold = float(high_water_threshold)
        self.medium_water_threshold = float(medium_water_threshold)
        self.current_step = 0

        # 奖励参数：强调“风险下降”，并兼顾能耗和启停成本
        self.overflow_weight = 12.0
        self.high_water_weight = 3.5
        self.medium_water_weight = 0.9
        self.risk_reduction_weight = 2.2
        self.energy_weight = 0.015
        self.switch_weight = 0.02
        self.idle_risk_penalty = 1.2
        self.low_risk_idle_bonus = 0.0
        self.rain_alert_threshold = 2.0
        self.rain_idle_penalty_weight = 0.15

        # 泵能力越大，开泵后对应片区水位下降越明显
        self.pump_capacity = np.array([1.5, 2.0, 3.0], dtype=np.float32)
        # 泵能力越大，单位动作能耗通常也更高
        self.pump_energy = np.array([1.0, 2.0, 3.5], dtype=np.float32)

        self.state: Optional[np.ndarray] = None
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        # 与 CSV 中实际泵数量对齐：默认 3 泵全开，读取 SWMM CSV 后会自动更新
        self.active_pump_count = 3
        self.pump_mask = np.ones(self.action_dim, dtype=np.float32)

        # SWMM 数据容器
        self.data_source = "random"
        self.swmm_series: Optional[Dict[str, Any]] = None
        self.data_len = 0
        self.data_cursor = 0
        if swmm_csv_path:
            self.load_swmm_csv(swmm_csv_path)

    def load_swmm_csv(self, csv_path: str) -> None:
        """
        读取 SWMM 导出的 CSV 数据。
        支持两种格式：
        1) 简化列：rain / water_1..3 / flow_1..3 / storage_1..3 / inflow(可选)
        2) SWMM宽表：system|... / node|... / link|...（自动映射）
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"SWMM csv not found: {csv_path}")

        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"SWMM csv is empty: {csv_path}")

        self.swmm_series = self._build_swmm_series(df)
        self.data_len = len(self.swmm_series["rain"])
        if self.data_len < 2:
            raise ValueError("SWMM csv requires at least 2 rows.")

        # 根据宽表中的泵流量列，自动推断“有效泵数量”（最多 3）
        flow_cols = self.swmm_series.get("_flow_cols", [])
        pump_flow_cols = [
            c
            for c in flow_cols
            if isinstance(c, str) and re.match(r"^link\|P\d+\|flow$", c)
        ]
        inferred_pumps = min(len(pump_flow_cols), self.action_dim)
        if inferred_pumps > 0:
            self.active_pump_count = inferred_pumps
        else:
            self.active_pump_count = self.action_dim
        self.pump_mask = np.zeros(self.action_dim, dtype=np.float32)
        self.pump_mask[: self.active_pump_count] = 1.0

        self.data_source = "swmm_csv"
        self.data_cursor = 0

    def _build_swmm_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        names = list(df.columns)

        def _to_arr(name: str) -> np.ndarray:
            s = pd.to_numeric(df[name], errors="coerce").fillna(0.0)
            return s.to_numpy(dtype=np.float32).reshape(-1)

        def _first_existing(candidates: list[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        def _pick_top_by_signal(candidates: list[str], k: int) -> list[str]:
            if not candidates:
                return []
            scored = []
            for c in candidates:
                s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                # 优先选择“有变化且量级较大”的时序，避免选到全零列
                score = float(s.std()) + 0.1 * float(s.abs().mean())
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:k]]

        def _stack_k(selected: list[str], k: int) -> np.ndarray:
            if not selected:
                return np.zeros((len(df), k), dtype=np.float32)
            arrs = [_to_arr(c) for c in selected]
            while len(arrs) < k:
                arrs.append(arrs[-1].copy())
            return np.stack(arrs[:k], axis=1).astype(np.float32)

        def _pump_sort_key(col: str) -> tuple[int, str]:
            # link|P12|flow -> 12；无法解析时放到末尾
            parts = col.split("|")
            if len(parts) >= 3:
                m = re.fullmatch(r"P(\d+)", parts[1])
                if m:
                    return (int(m.group(1)), col)
            return (10**9, col)

        compact_required = [
            "rain",
            "water_1",
            "water_2",
            "water_3",
            "flow_1",
            "flow_2",
            "flow_3",
            "storage_1",
            "storage_2",
            "storage_3",
        ]
        has_compact = all(c in df.columns for c in compact_required)

        if has_compact:
            rain = _to_arr("rain")
            water = np.stack(
                [_to_arr("water_1"), _to_arr("water_2"), _to_arr("water_3")], axis=1
            )
            flow = np.stack(
                [_to_arr("flow_1"), _to_arr("flow_2"), _to_arr("flow_3")], axis=1
            )
            storage = np.stack(
                [_to_arr("storage_1"), _to_arr("storage_2"), _to_arr("storage_3")],
                axis=1,
            )
            inflow_col = _first_existing(
                ["inflow", "system|lateral_inflow", "system|direct_inflow"]
            )
            inflow = (
                _to_arr(inflow_col)
                if inflow_col is not None
                else np.zeros(len(rain), dtype=np.float32)
            )
            return {
                "rain": rain,
                "water": water.astype(np.float32),
                "flow": flow.astype(np.float32),
                "storage": storage.astype(np.float32),
                "inflow": inflow,
                "_rain_col": "rain",
                "_water_cols": ["water_1", "water_2", "water_3"],
                "_flow_cols": ["flow_1", "flow_2", "flow_3"],
                "_storage_cols": ["storage_1", "storage_2", "storage_3"],
                "_inflow_col": inflow_col,
            }

        # 宽表自动映射（适配 system|/node|/link|）
        rain_col = _first_existing(
            [
                "system|rainfall",
                "rain",
                "subcatchment|S2|rainfall",
                "subcatchment|S1|rainfall",
            ]
        )
        if rain_col is None:
            # 兼容极简训练 CSV：允许无雨量列，默认按 0 降雨处理
            rain = np.zeros(len(df), dtype=np.float32)
        else:
            rain = _to_arr(rain_col)

        node_depth_cols = [c for c in names if c.startswith("node|") and c.endswith("|depth")]
        node_volume_cols = [c for c in names if c.startswith("node|") and c.endswith("|volume")]
        link_flow_cols = [c for c in names if c.startswith("link|") and c.endswith("|flow")]

        if not node_depth_cols:
            raise ValueError("CSV无法识别节点水位列（期望包含 'node|*|depth'）。")

        top_depth_cols = _pick_top_by_signal(node_depth_cols, 3)
        top_volume_cols = _pick_top_by_signal(node_volume_cols, 3)

        pump_flow_cols = [
            c for c in link_flow_cols if re.match(r"^link\|P\d+\|flow$", c)
        ]
        pump_flow_cols = sorted(pump_flow_cols, key=_pump_sort_key)
        top_flow_cols = list(pump_flow_cols[:3])
        if len(top_flow_cols) < 3:
            for c in _pick_top_by_signal(link_flow_cols, 6):
                if c not in top_flow_cols:
                    top_flow_cols.append(c)
                if len(top_flow_cols) >= 3:
                    break
        if len(top_flow_cols) < 3:
            outflow_col = _first_existing(["system|outflow"])
            if outflow_col is not None:
                top_flow_cols.append(outflow_col)

        water = _stack_k(top_depth_cols, 3)
        flow = _stack_k(top_flow_cols, 3)
        # 轻量 rl_core CSV 可能不包含 node|*|volume。
        # 此时用 depth 近似构造 storage 代理特征，避免状态尾部全零。
        if top_volume_cols:
            storage = _stack_k(top_volume_cols, 3)
        else:
            storage = np.clip(water * 35.0, 0.0, 100.0).astype(np.float32)

        inflow_col = _first_existing(
            [
                "system|lateral_inflow",
                "system|direct_inflow",
                "system|dry_weather_inflow",
                "system|groundwater_inflow",
                "inflow",
            ]
        )
        inflow = (
            _to_arr(inflow_col)
            if inflow_col is not None
            else np.zeros(len(rain), dtype=np.float32)
        )

        return {
            "rain": rain,
            "water": water.astype(np.float32),
            "flow": flow.astype(np.float32),
            "storage": storage.astype(np.float32),
            "inflow": inflow,
            "_rain_col": rain_col,
            "_water_cols": top_depth_cols,
            "_flow_cols": top_flow_cols,
            "_storage_cols": top_volume_cols,
            "_inflow_col": inflow_col,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        # 每个 episode 开始时重置时间和上一时刻动作
        self.current_step = 0
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        if self.data_source == "swmm_csv" and self.swmm_series is not None:
            # 支持从任意时间索引启动（便于离线数据训练/评估）
            if "start_idx" in options:
                start_idx = int(options.get("start_idx", 0))
            elif self.random_start:
                start_idx = int(self.np_random.integers(0, self.data_len - 1))
            else:
                start_idx = 0
            start_idx = int(np.clip(start_idx, 0, self.data_len - 2))
            self.data_cursor = start_idx

            self.state = self._compose_state_from_swmm(self.data_cursor)
            info = {
                "data_source": self.data_source,
                "cursor": self.data_cursor,
                "data_len": self.data_len,
            }
            return self.state.copy(), info

        # random 模式：随机初始化一个“合理”的系统状态
        rain = float(self.np_random.uniform(0.0, 8.0))
        water = self.np_random.uniform(1.0, 6.0, size=3).astype(np.float32)
        flow = self.np_random.uniform(2.0, 10.0, size=3).astype(np.float32)
        storage = self.np_random.uniform(20.0, 70.0, size=3).astype(np.float32)
        self.state = np.concatenate([[rain], water, flow, storage]).astype(np.float32)
        obs_box = cast(spaces.Box, self.observation_space)
        self.state = np.clip(self.state, obs_box.low, obs_box.high)
        return self.state.copy(), {"data_source": self.data_source}

    def _compose_state_from_swmm(self, idx: int) -> np.ndarray:
        if self.swmm_series is None:
            raise RuntimeError("SWMM series is not loaded.")
        rain = self.swmm_series["rain"][idx]
        water = self.swmm_series["water"][idx]
        flow = self.swmm_series["flow"][idx]
        storage = self.swmm_series["storage"][idx]
        state = np.concatenate([[rain], water, flow, storage]).astype(np.float32)
        obs_box = cast(spaces.Box, self.observation_space)
        return np.clip(state, obs_box.low, obs_box.high)

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        self.current_step += 1
        # 兼容 list/ndarray 输入，并强制映射到 {0,1}
        action = np.asarray(action).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} actions, got {action.shape}")
        action = np.clip(np.rint(action), 0, 1).astype(np.float32)
        if self.data_source == "swmm_csv":
            action = action * self.pump_mask

        prev_state = self.state.copy()
        rain, water_levels, flow, storage = self._transition(prev_state, action)

        next_state = np.concatenate([[rain], water_levels, flow, storage]).astype(
            np.float32
        )
        obs_box = cast(spaces.Box, self.observation_space)
        next_state = np.clip(next_state, obs_box.low, obs_box.high).astype(np.float32)

        # 风险指标（overflow/high/medium）+ 成本指标（energy/switch）
        prev_overflow, prev_high, prev_medium, prev_risk = self._risk_metrics(
            prev_state[1:4]
        )
        overflow, high_water, medium_water, current_risk = self._risk_metrics(
            water_levels
        )
        energy = float(np.sum(action * self.pump_energy))
        switch_penalty = float(np.sum(np.abs(action - self.last_action)))

        # 奖励设计：
        # 1) 当前风险越高，惩罚越大
        # 2) 相比上一时刻风险下降则给正向激励（引导“有效开泵”）
        # 3) 能耗与频繁启停惩罚
        # 4) 在有风险或强降雨前兆时，不作为会被额外惩罚
        reward = 0.0
        reward -= current_risk
        reward += (prev_risk - current_risk) * self.risk_reduction_weight
        reward -= energy * self.energy_weight
        reward -= switch_penalty * self.switch_weight

        pump_count = float(np.sum(action))
        max_water = float(np.max(water_levels))
        if pump_count == 0.0:
            if max_water > self.high_water_threshold:
                reward -= self.idle_risk_penalty
            if rain > self.rain_alert_threshold:
                reward -= (
                    rain - self.rain_alert_threshold
                ) * self.rain_idle_penalty_weight
            if max_water < self.medium_water_threshold:
                reward += self.low_risk_idle_bonus

        terminated = False
        truncated = False

        # 严重溢流直接终止当前 episode，模拟调度失败
        if overflow > self.severe_overflow_threshold:
            terminated = True
            reward -= 20.0

        # 达到最大步长则自然截断
        if self.current_step >= self.max_steps:
            truncated = True

        # 限制 reward 数值范围，避免值函数学习不稳定
        reward = float(np.clip(reward, -200.0, 50.0))

        self.state = next_state
        self.last_action = action.copy()

        info = {
            "data_source": self.data_source,
            "active_pump_count": int(self.active_pump_count),
            "pump_mask": self.pump_mask.astype(np.float32).tolist(),
            "rain": float(rain),
            "overflow": overflow,
            "high_water_penalty": high_water,
            "medium_water_penalty": medium_water,
            "risk": current_risk,
            "risk_reduction": float(prev_risk - current_risk),
            "prev_overflow": prev_overflow,
            "prev_high_water_penalty": prev_high,
            "prev_medium_water_penalty": prev_medium,
            "energy": energy,
            "switch_penalty": switch_penalty,
            "reward": reward,
        }
        if self.data_source == "swmm_csv":
            info["cursor"] = self.data_cursor
            info["data_len"] = self.data_len

        return next_state.copy(), reward, terminated, truncated, info

    def _risk_metrics(self, water_levels: np.ndarray):
        overflow = float(np.sum(np.maximum(water_levels - 8.0, 0.0)))
        high_water = float(
            np.sum(np.maximum(water_levels - self.high_water_threshold, 0.0))
        )
        medium_water = float(
            np.sum(np.maximum(water_levels - self.medium_water_threshold, 0.0))
        )
        risk = (
            overflow * self.overflow_weight
            + high_water * self.high_water_weight
            + medium_water * self.medium_water_weight
        )
        return overflow, high_water, medium_water, risk

    def _transition(self, prev_state: np.ndarray, action: np.ndarray):
        # 统一入口：按数据源选择状态转移逻辑
        if self.data_source == "swmm_csv" and self.swmm_series is not None:
            return self._transition_from_swmm(prev_state, action)
        return self._transition_random(prev_state, action)

    def _transition_random(self, prev_state: np.ndarray, action: np.ndarray):
        # 构造“有周期 + 偶发暴雨”的降雨过程
        rain = 5.0 + 2.5 * np.sin(self.current_step / 15.0)
        if self.np_random.random() < 0.05:
            rain += float(self.np_random.uniform(3.0, 8.0))
        rain = max(rain, 0.0)

        pump_effect = action * self.pump_capacity
        total_pump_effect = float(np.sum(pump_effect))

        current_water = prev_state[1:4]
        # 蓄积越高，对管网水位抬升的压力越大
        catchment_pressure = float(np.mean(prev_state[7:10])) / 100.0
        water_levels = (
            current_water * 0.90
            + rain * 0.35
            + catchment_pressure * 0.5
            - pump_effect * 2.8
        )
        water_levels += self.np_random.normal(0.0, 0.05, size=3).astype(np.float32)
        water_levels = np.clip(water_levels, 0.0, 12.0).astype(np.float32)

        current_flow = prev_state[4:7]
        # 泵开得越多，流量会增加（排水通道负荷提高）
        flow = current_flow * 0.90 + rain * 0.25 + total_pump_effect * 0.20
        flow = np.clip(flow, 0.0, 20.0).astype(np.float32)

        current_storage = prev_state[7:10]
        # storage 可理解为系统“蓄积压力”，降雨增加、排水减少
        storage = current_storage * 0.98 + rain * 0.20 - total_pump_effect * 0.35
        storage = np.clip(storage, 0.0, 100.0).astype(np.float32)

        return float(rain), water_levels, flow, storage

    def _transition_from_swmm(self, prev_state: np.ndarray, action: np.ndarray):
        if self.swmm_series is None:
            raise RuntimeError("SWMM series is not loaded.")

        next_idx = min(self.data_cursor + 1, self.data_len - 1)
        self.data_cursor = next_idx

        rain = float(self.swmm_series["rain"][next_idx])
        target_water = self.swmm_series["water"][next_idx]
        target_flow = self.swmm_series["flow"][next_idx]
        target_storage = self.swmm_series["storage"][next_idx]
        inflow = float(self.swmm_series["inflow"][next_idx])

        pump_effect = action * self.pump_capacity
        total_pump_effect = float(np.sum(pump_effect))

        # 使用 SWMM 结果作为基准轨迹，叠加动作扰动：
        # target_* 负责“真实趋势”，action 负责“调度影响”。
        water_levels = (
            target_water + 0.20 * prev_state[1:4] - 2.00 * pump_effect + 0.05 * inflow
        )
        water_levels = np.clip(water_levels, 0.0, 12.0).astype(np.float32)

        flow = target_flow + 0.20 * total_pump_effect + 0.08 * inflow
        flow = np.clip(flow, 0.0, 20.0).astype(np.float32)

        storage = target_storage - 0.40 * total_pump_effect + 0.06 * inflow
        storage = np.clip(storage, 0.0, 100.0).astype(np.float32)

        return rain, water_levels, flow, storage

    def render(self):
        print(
            f"step={self.current_step} source={self.data_source} "
            f"state={None if self.state is None else np.round(self.state, 3)}"
        )
