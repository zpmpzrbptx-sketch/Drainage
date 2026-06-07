import os
import json
import importlib.util
from pathlib import Path
from typing import cast
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import sys

# 获取项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加入Python搜索路径
sys.path.append(ROOT_DIR)

# 导入你自己的环境
from env.drainage_env import DrainageEnv

# =========================
# 1. 创建目录
# =========================
LOG_DIR = "logs/"
MODEL_DIR = "models/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# 2. 创建环境
# =========================
SWMM_CSV_PATH = os.getenv("SWMM_CSV_PATH", "").strip()
SWMM_CSV_PATHS = os.getenv("SWMM_CSV_PATHS", "").strip()
SWMM_DATA_DIR = os.getenv("SWMM_DATA_DIR", "").strip()
TRAIN_MODE = os.getenv("TRAIN_MODE", "single_m0").strip().lower()
TRAIN_START_MODE = os.getenv("TRAIN_START_MODE", "").strip().lower()
MODEL_BASENAME = os.getenv("MODEL_BASENAME", "").strip()
# 单文件：
# export SWMM_CSV_PATH=/path/to/swmm_export.csv
# 多文件（推荐用于“多个csv同步训练”）：
# export SWMM_CSV_PATHS=/path/a.csv,/path/b.csv
# 数据目录（可选）：
# export SWMM_DATA_DIR=data/processed_rl_core
# 常用模式：
# export TRAIN_MODE=single_m0
# export TRAIN_MODE=single_s0
# export TRAIN_MODE=single_s1
# export TRAIN_MODE=single_s2
# export TRAIN_MODE=multi_storm
# export TRAIN_MODE=multi_all
# export MODEL_BASENAME=ppo_drainage_custom
# 可选起点模式：
# export TRAIN_START_MODE=risk_weighted


def _resolve_processed_dir() -> Path:
    if SWMM_DATA_DIR:
        custom = Path(SWMM_DATA_DIR)
        if not custom.is_absolute():
            custom = Path(ROOT_DIR) / custom
        return custom

    # 兼容新旧目录：优先 rl_core，再回退 processed
    candidates = [
        Path(ROOT_DIR) / "data" / "processed_rl_core",
        Path(ROOT_DIR) / "data" / "processed",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _slugify_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "default"


def resolve_model_stem() -> str:
    if MODEL_BASENAME:
        return _slugify_name(MODEL_BASENAME)
    return f"ppo_drainage_{_slugify_name(TRAIN_MODE)}"


def resolve_swmm_csv_paths() -> list[str]:
    processed_dir = _resolve_processed_dir()

    def _discover_csv(keyword: str | None = None) -> list[str]:
        if not processed_dir.exists():
            return []
        files = sorted(processed_dir.glob("*.csv"))
        if keyword is None:
            return [str(p) for p in files]
        key = keyword.lower()
        return [str(p) for p in files if key in p.stem.lower()]

    def _discover_any(keywords: list[str]) -> list[str]:
        for keyword in keywords:
            paths = _discover_csv(keyword)
            if paths:
                return paths
        return []

    # 优先使用显式列表变量
    if SWMM_CSV_PATHS:
        paths = [p.strip() for p in SWMM_CSV_PATHS.split(",") if p.strip()]
        if paths:
            return paths

    # 兼容原有单文件变量
    if SWMM_CSV_PATH:
        return [SWMM_CSV_PATH]

    # 实验开关：single_m0 / single_s0 / multi_storm / multi_all
    if TRAIN_MODE in {"single_m0", "single_moderate", "single_m0_only"}:
        m0_paths = _discover_csv("moderaterain_m0")
        if m0_paths:
            return [m0_paths[0]]

    if TRAIN_MODE == "multi_m0":
        m0_paths = _discover_csv("moderaterain_m0")
        if m0_paths:
            return m0_paths

    if TRAIN_MODE in {"single_m1", "single_m1_only"}:
        m1_paths = _discover_csv("moderaterain_m1")
        if m1_paths:
            return [m1_paths[0]]

    if TRAIN_MODE == "multi_m1":
        m1_paths = _discover_csv("moderaterain_m1")
        if m1_paths:
            return m1_paths

    if TRAIN_MODE in {"single_m2", "single_m2_only"}:
        m2_paths = _discover_csv("moderaterain_m2")
        if m2_paths:
            return [m2_paths[0]]

    if TRAIN_MODE == "multi_m2":
        m2_paths = _discover_csv("moderaterain_m2")
        if m2_paths:
            return m2_paths

    if TRAIN_MODE in {"single_s0", "single_s0_only"}:
        s0_paths = _discover_any(["torrentialrain_s0", "storm_s0", "s0"])
        if s0_paths:
            return [s0_paths[0]]

    if TRAIN_MODE == "multi_s0":
        s0_paths = _discover_any(["torrentialrain_s0", "storm_s0", "s0"])
        if s0_paths:
            return s0_paths

    if TRAIN_MODE in {"single_s1", "single_s1_only"}:
        s1_paths = _discover_any(["torrentialrain_s1", "storm_s1", "s1"])
        if s1_paths:
            return [s1_paths[0]]

    if TRAIN_MODE == "multi_s1":
        s1_paths = _discover_any(["torrentialrain_s1", "storm_s1", "s1"])
        if s1_paths:
            return s1_paths

    if TRAIN_MODE in {"single_s2", "single_s2_only"}:
        s2_paths = _discover_any(["torrentialrain_s2", "storm_s2", "s2"])
        if s2_paths:
            return [s2_paths[0]]

    if TRAIN_MODE == "multi_s2":
        s2_paths = _discover_any(["torrentialrain_s2", "storm_s2", "s2"])
        if s2_paths:
            return s2_paths

    if TRAIN_MODE in {"single_s", "single_storm"}:
        s2_paths = _discover_csv("s2")
        if s2_paths:
            return [s2_paths[0]]

    if TRAIN_MODE in {"multi_s", "multi_storm", "multi_torrential"}:
        storm_paths = []
        for p in _discover_csv():
            stem = Path(p).stem.lower()
            if "torrentialrain_" in stem or "_storm_" in stem:
                storm_paths.append(p)
        if storm_paths:
            return storm_paths

    if TRAIN_MODE in {"single_all", "single"}:
        all_paths = _discover_csv()
        if all_paths:
            return [all_paths[0]]

    # 默认与兜底：使用目录下全部 csv（适配当前 processed_rl_core）
    all_paths = _discover_csv()
    if all_paths:
        return all_paths

    return []


TRAIN_SWMM_CSVS = resolve_swmm_csv_paths()
MODEL_STEM = resolve_model_stem()
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_STEM)
MODEL_ZIP_PATH = f"{MODEL_SAVE_PATH}.zip"
MODEL_META_SAVE_PATH = os.path.join(MODEL_DIR, f"{MODEL_STEM}.meta.json")
HAS_TENSORBOARD = importlib.util.find_spec("tensorboard") is not None
HAS_TQDM = importlib.util.find_spec("tqdm") is not None
HAS_RICH = importlib.util.find_spec("rich") is not None
HAS_PROGRESS_BAR = HAS_TQDM and HAS_RICH
print(f"数据目录: {_resolve_processed_dir()}")
if TRAIN_SWMM_CSVS:
    print(f"训练模式: {TRAIN_MODE}")
    print("训练使用 SWMM CSV:")
    for p in TRAIN_SWMM_CSVS:
        print(f"- {p}")
else:
    print("未检测到 SWMM CSV，回退到 random 模式训练。")
print(f"模型保存标识: {MODEL_STEM}")
if not HAS_TENSORBOARD:
    print("[提示] 未检测到 tensorboard，训练将继续，但不写入 TensorBoard 日志。")
if not HAS_PROGRESS_BAR:
    print("[提示] 未检测到 tqdm 或 rich，训练将继续，但不显示进度条。")


class NormalizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: DrainageEnv):
        super().__init__(env)
        self.observation_space = env.get_normalized_observation_space()

    def observation(self, observation):
        drainage_env = cast(DrainageEnv, self.env)
        return drainage_env.normalize_observation(observation)


def inspect_csv_signal(csv_path: str) -> dict:
    # 复用环境中的 CSV 映射逻辑，兼容简化列和 SWMM 宽表列
    env = DrainageEnv(swmm_csv_path=csv_path, random_start=False)
    if env.swmm_series is None:
        raise ValueError(f"{csv_path} parsed but swmm_series is empty")

    rain = env.swmm_series["rain"].astype(float)
    water_max = env.swmm_series["water"].astype(float).max(axis=1)
    flow_abs_max = abs(env.swmm_series["flow"].astype(float)).max(axis=1)
    storage_max = env.swmm_series["storage"].astype(float).max(axis=1)

    return {
        "path": csv_path,
        "rows": int(len(rain)),
        "rain_nonzero_ratio": float((rain > 1e-6).mean()),
        "rain_max": float(rain.max()),
        "water_peak": float(water_max.max()),
        "water_std": float(water_max.std()),
        "flow_abs_peak": float(flow_abs_max.max()),
        "storage_peak": float(storage_max.max()),
    }


def is_weak_signal(stats: dict) -> bool:
    return (
        stats["rain_nonzero_ratio"] < 0.01
        and stats["rain_max"] < 0.1
        and stats["water_peak"] < 0.5
        and stats["flow_abs_peak"] < 0.5
        and stats["storage_peak"] < 0.5
    )


CSV_STATS = []
if TRAIN_SWMM_CSVS:
    print("训练数据体检:")
    for p in TRAIN_SWMM_CSVS:
        st = inspect_csv_signal(p)
        CSV_STATS.append(st)
        print(
            f"- {Path(p).name}: rows={st['rows']}, rain_nonzero={st['rain_nonzero_ratio']:.3f}, "
            f"rain_max={st['rain_max']:.3f}, water_peak={st['water_peak']:.3f}, "
            f"flow_abs_peak={st['flow_abs_peak']:.3f}, storage_peak={st['storage_peak']:.3f}"
        )

WEAK_DATASET = bool(CSV_STATS) and all(is_weak_signal(s) for s in CSV_STATS)
if WEAK_DATASET:
    print("[警告] 当前 SWMM CSV 全部为弱信号（近零降雨/近零水位）。将混入 random 环境以避免策略假收敛。")


def make_env(rank: int = 0):
    def _init():
        # 每个并行环境都创建独立实例，避免状态互相污染
        csv_path = None
        use_random_env = WEAK_DATASET and (rank % 4 == 0)
        default_start_mode = "risk_weighted" if TRAIN_MODE in {"single_m2", "single_m2_only", "multi_m2"} else "uniform"
        start_mode = TRAIN_START_MODE or default_start_mode
        if TRAIN_SWMM_CSVS and not use_random_env:
            # 多环境轮询分配数据源，实现多 CSV 并行“同步训练”
            csv_path = TRAIN_SWMM_CSVS[rank % len(TRAIN_SWMM_CSVS)]

        env = DrainageEnv(
            swmm_csv_path=csv_path,
            random_start=True,
            random_start_mode=start_mode,
        )
        env = NormalizedObservationWrapper(env)
        # Monitor 会记录 episode reward/length，便于 SB3 日志统计
        env = Monitor(env)
        # 不同 rank 使用不同 seed，提升采样多样性
        env.reset(seed=42 + rank)
        return env

    return _init


N_ENVS = 8
# 并行环境：提高采样效率，让策略更快看到多场景数据
env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])


# =========================
# 3. 创建模型（PPO）
# =========================
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR if HAS_TENSORBOARD else None,
    # 这组参数偏“稳定优先”：
    # - learning_rate 降低，减少策略震荡
    # - n_steps * N_ENVS = 4096，每轮 rollout 数据量更充足
    # - batch_size 调大，价值网络更新更平滑
    # - ent_coef 促进探索，避免过早塌缩到单一动作
    learning_rate=2e-4,
    n_steps=512,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.12,
    ent_coef=0.06,
    vf_coef=0.7,
)


# =========================
# 4. 开始训练
# =========================
TOTAL_TIMESTEPS = 1_000_000
# 复杂控制任务通常需要更长训练步数，5万步常常不够收敛

print("开始训练...")
print(f"起点采样模式: {TRAIN_START_MODE or ('risk_weighted' if TRAIN_MODE in {'single_m2', 'single_m2_only', 'multi_m2'} else 'uniform')}")

model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=HAS_PROGRESS_BAR)

print("训练完成！")


# =========================
# 5. 保存模型
# =========================
model.save(MODEL_SAVE_PATH)

with open(MODEL_META_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "model_stem": MODEL_STEM,
            "model_path": MODEL_ZIP_PATH,
            "meta_path": MODEL_META_SAVE_PATH,
            "obs_normalized": True,
            "train_mode": TRAIN_MODE,
            "train_start_mode": TRAIN_START_MODE or ("risk_weighted" if TRAIN_MODE in {"single_m2", "single_m2_only", "multi_m2"} else "uniform"),
            "total_timesteps": TOTAL_TIMESTEPS,
            "csv_paths": TRAIN_SWMM_CSVS,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"模型已保存到: {MODEL_ZIP_PATH}")
print(f"模型元数据已保存到: {MODEL_META_SAVE_PATH}")


# =========================
# 6. 简单测试（训练后）
# =========================
print("开始测试模型...")

test_csv_path = TRAIN_SWMM_CSVS[0] if TRAIN_SWMM_CSVS else None
test_env = DrainageEnv(swmm_csv_path=test_csv_path, random_start=True)
obs, _ = test_env.reset()

print("随机动作对照（20步）...")
rand_env = DrainageEnv(swmm_csv_path=test_csv_path, random_start=True)
rand_obs, _ = rand_env.reset()
for step in range(20):
    rand_action = rand_env.action_space.sample()
    rand_obs, rand_reward, rand_done, rand_truncated, _ = rand_env.step(rand_action)
    print(f"[Random] Step {step} | Action: {rand_action} | Reward: {rand_reward:.2f}")
    if rand_done or rand_truncated:
        rand_obs, _ = rand_env.reset()

for step in range(50):
    # deterministic=True 表示评估阶段用“贪心动作”，不加探索噪声
    model_obs = test_env.normalize_observation(obs) if hasattr(test_env, "normalize_observation") else obs
    action, _states = model.predict(model_obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)

    max_water = float(max(obs[1:4]))
    overflow = float(info.get("overflow", 0.0))
    risk = float(info.get("risk", 0.0))

    print(
        f"Step {step} | Action: {action} | max_water: {max_water:.2f} "
        f"| overflow: {overflow:.2f} | risk: {risk:.2f} | Reward: {reward:.2f}"
    )

    if done or truncated:
        # 测试时随机采样起点，避免只落在平稳时段
        reset_options = None
        if test_env.data_source == "swmm_csv" and test_env.data_len > 2:
            start_idx = int(test_env.np_random.integers(0, test_env.data_len - 1))
            reset_options = {"start_idx": start_idx}
        obs, _ = test_env.reset(options=reset_options)

if test_env.data_source == "swmm_csv" and WEAK_DATASET:
    print("[提示] 当前测试数据为弱信号片段（risk≈0），全关泵是合理策略，不代表模型退化。")

print("测试完成")
