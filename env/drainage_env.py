from pathlib import Path
from typing import Any, Dict, Optional, cast

import gymnasium as gym
import numpy as np
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
        self.current_step = 0

        # 奖励参数（历史保留字段，当前 reward 直接写在 step 中）
        self.alpha = 8.0  # 溢流惩罚
        self.beta = 0.3  # 能耗惩罚
        self.gamma = 2.0  # 高水位惩罚

        # 泵能力越大，开泵后对应片区水位下降越明显
        self.pump_capacity = np.array([1.5, 2.0, 3.0], dtype=np.float32)
        # 泵能力越大，单位动作能耗通常也更高
        self.pump_energy = np.array([1.0, 2.0, 3.5], dtype=np.float32)

        self.state: Optional[np.ndarray] = None
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        # SWMM 数据容器
        self.data_source = "random"
        self.swmm_series: Optional[Dict[str, np.ndarray]] = None
        self.data_len = 0
        self.data_cursor = 0
        if swmm_csv_path:
            self.load_swmm_csv(swmm_csv_path)

    def load_swmm_csv(self, csv_path: str) -> None:
        """
        读取 SWMM 导出的 CSV 数据。
        必需列：
        - rain
        - water_1, water_2, water_3
        - flow_1, flow_2, flow_3
        - storage_1, storage_2, storage_3
        可选列：
        - inflow（用于增强随机动力学）
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"SWMM csv not found: {csv_path}")

        raw = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
        if raw.size == 0:
            raise ValueError(f"SWMM csv is empty: {csv_path}")
        # 统一转成一维记录数组，兼容“只有1行数据”时的形状
        data = np.atleast_1d(raw).view(np.recarray)
        names = data.dtype.names
        if names is None:
            raise ValueError(f"SWMM csv has no named columns: {csv_path}")

        required = [
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
        missing = [c for c in required if c not in names]
        if missing:
            raise ValueError(
                f"SWMM csv missing columns: {missing}. "
                f"Required columns are: {required}"
            )

        def col(name: str) -> np.ndarray:
            return np.asarray(cast(Any, data)[name], dtype=np.float32).reshape(-1)

        # 转为模型更方便使用的结构：
        # rain:(T,), water/flow/storage:(T,3), inflow:(T,)
        self.swmm_series = {
            "rain": col("rain"),
            "water": np.stack([col("water_1"), col("water_2"), col("water_3")], axis=1),
            "flow": np.stack([col("flow_1"), col("flow_2"), col("flow_3")], axis=1),
            "storage": np.stack(
                [col("storage_1"), col("storage_2"), col("storage_3")], axis=1
            ),
            "inflow": (
                col("inflow")
                if "inflow" in names
                else np.zeros(len(col("rain")), dtype=np.float32)
            ),
        }
        self.data_len = len(self.swmm_series["rain"])
        if self.data_len < 2:
            raise ValueError("SWMM csv requires at least 2 rows.")

        self.data_source = "swmm_csv"
        self.data_cursor = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        # 每个 episode 开始时重置时间和上一时刻动作
        self.current_step = 0
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        if self.data_source == "swmm_csv" and self.swmm_series is not None:
            # 支持从任意时间索引启动（便于离线数据训练/评估）
            start_idx = int(options.get("start_idx", 0))
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

        prev_state = self.state.copy()
        rain, water_levels, flow, storage = self._transition(prev_state, action)

        next_state = np.concatenate([[rain], water_levels, flow, storage]).astype(
            np.float32
        )
        obs_box = cast(spaces.Box, self.observation_space)
        next_state = np.clip(next_state, obs_box.low, obs_box.high).astype(np.float32)

        # 关键指标：
        # overflow: 超过 8.0 的总超高量（越大越危险）
        overflow = float(np.sum(np.maximum(water_levels - 8.0, 0.0)))
        high_water = float(np.sum(water_levels > 6.0))
        energy = float(np.sum(action * self.pump_energy))
        low_water_penalty = float(np.sum(water_levels < 1.0))
        switch_penalty = float(np.sum(np.abs(action - self.last_action)))

        prev_water = float(np.mean(prev_state[1:4]))
        new_water = float(np.mean(water_levels))
        target_water = 3.5

        # 奖励设计目标：维持安全水位，而不是“总是关泵”或“总是开泵”。
        # 下面的项本质上是“安全性 + 经济性 + 平稳性”的加权和。
        reward = 0.0

        # 溢流惩罚（核心）
        reward -= overflow * 4.0

        # 高水位惩罚
        reward -= high_water * 1.5

        # 能耗惩罚（轻）
        reward -= energy * 0.1

        # 频繁切换惩罚
        reward -= switch_penalty * 0.1

        # 低水位惩罚
        reward -= low_water_penalty * 1.0

        # 偏离目标水位
        reward -= abs(new_water - target_water) * 1.2

        # 水位下降奖励
        reward += (prev_water - new_water) * 6.0

        # 安全运行奖励
        if overflow == 0.0 and 2.0 <= new_water <= 5.0:
            reward += 5.0

        if overflow == 0.0 and 2.0 <= new_water <= 5.0:
            reward += 2.0

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
            "rain": float(rain),
            "overflow": overflow,
            "energy": energy,
            "high_water": high_water,
            "switch_penalty": switch_penalty,
            "reward": reward,
        }
        if self.data_source == "swmm_csv":
            info["cursor"] = self.data_cursor
            info["data_len"] = self.data_len

        return next_state.copy(), reward, terminated, truncated, info

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
            target_water + 0.20 * prev_state[1:4] - 0.80 * pump_effect + 0.05 * inflow
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
