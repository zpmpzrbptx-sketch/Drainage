import os
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
# 可通过环境变量切换到真实 SWMM 数据：
# export SWMM_CSV_PATH=/path/to/swmm_export.csv


def make_env(rank: int = 0):
    def _init():
        # 每个并行环境都创建独立实例，避免状态互相污染
        env = DrainageEnv(swmm_csv_path=SWMM_CSV_PATH or None)
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
    tensorboard_log=LOG_DIR,
    # 这组参数偏“稳定优先”：
    # - learning_rate 降低，减少策略震荡
    # - n_steps * N_ENVS = 4096，每轮 rollout 数据量更充足
    # - batch_size 调大，价值网络更新更平滑
    # - ent_coef 促进探索，避免过早塌缩到单一动作
    learning_rate=1e-4,
    n_steps=512,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.01,
    vf_coef=0.7,
)


# =========================
# 4. 开始训练
# =========================
TOTAL_TIMESTEPS = 200_000
# 复杂控制任务通常需要更长训练步数，5万步常常不够收敛

print("开始训练...")

model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

print("训练完成！")


# =========================
# 5. 保存模型
# =========================
model_path = os.path.join(MODEL_DIR, "ppo_drainage")

model.save(model_path)

print(f"模型已保存到: {model_path}")


# =========================
# 6. 简单测试（训练后）
# =========================
print("开始测试模型...")

test_env = DrainageEnv(swmm_csv_path=SWMM_CSV_PATH or None)
obs, _ = test_env.reset()

for step in range(50):
    # deterministic=True 表示评估阶段用“贪心动作”，不加探索噪声
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)

    print(f"Step {step} | Action: {action} | Reward: {reward:.2f}")

    if done:
        obs, _ = test_env.reset()

print("测试完成")
