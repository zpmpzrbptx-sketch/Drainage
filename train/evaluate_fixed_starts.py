import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from env.drainage_env import DrainageEnv  # noqa: E402

try:
    from stable_baselines3 import PPO  # noqa: E402
except Exception:  # pragma: no cover
    PPO = None


DEFAULT_MODEL_STEM = "ppo_drainage"


def _slugify_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "default"


def _resolve_default_model_stem() -> str:
    configured = os.getenv("EVAL_MODEL_BASENAME", "").strip()
    if configured:
        return _slugify_name(configured)

    train_mode = (
        os.getenv("EVAL_TRAIN_MODE", "").strip().lower()
        or os.getenv("TRAIN_MODE", "").strip().lower()
    )
    if train_mode:
        return f"ppo_drainage_{_slugify_name(train_mode)}"
    return DEFAULT_MODEL_STEM


def _resolve_default_model_paths() -> tuple[Path, Path]:
    stem = _resolve_default_model_stem()
    return (
        ROOT_DIR / "models" / f"{stem}.zip",
        ROOT_DIR / "models" / f"{stem}.meta.json",
    )


def _load_model_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _resolve_default_csv(meta_path: Path) -> str | None:
    meta = _load_model_meta(meta_path)
    train_mode = str(meta.get("train_mode", "")).strip().lower()
    csv_paths = meta.get("csv_paths", [])
    candidates: list[str]

    if isinstance(csv_paths, list):
        for value in csv_paths:
            if isinstance(value, str):
                path = Path(value)
                if path.exists():
                    return str(path)

    if train_mode.startswith("single_m0") or train_mode.startswith("multi_m0"):
        candidates = [
            "data/processed_rl_core/moderateRain_M0_medium_pre.csv",
            "data/processed/moderateRain_M0_medium_pre.csv",
        ]
    elif train_mode.startswith("single_m1") or train_mode.startswith("multi_m1"):
        candidates = [
            "data/processed_rl_core/moderateRain_M1_medium_water_level.csv",
            "data/processed/moderateRain_M1_medium_water_level.csv",
        ]
    elif train_mode.startswith("single_m2") or train_mode.startswith("multi_m2"):
        candidates = [
            "data/processed_rl_core/moderateRain_M2_medium_fsn_storage.csv",
            "data/processed/moderateRain_M2_medium_fsn_storage.csv",
        ]
    elif train_mode.startswith("single_s0") or train_mode.startswith("multi_s0"):
        candidates = [
            "data/processed_rl_core/torrentialRain_S0_storm_pre_realistic.csv",
            "data/processed/torrentialRain_S0_storm_pre_realistic.csv",
        ]
    elif train_mode.startswith("single_s1") or train_mode.startswith("multi_s1"):
        candidates = [
            "data/processed_rl_core/torrentialRain_S1_storm_12h_predrain_plant.csv",
            "data/processed/torrentialRain_S1_storm_12h_predrain_plant.csv",
        ]
    elif (
        train_mode.startswith("single_s2")
        or train_mode.startswith("multi_s2")
        or train_mode.startswith("single_s")
        or train_mode.startswith("multi_s")
        or "storm" in train_mode
        or "torrential" in train_mode
    ):
        candidates = [
            "data/processed_rl_core/torrentialRain_S2_storm_12h_fsn_storage.csv",
            "data/processed/torrentialRain_S2_storm_12h_fsn_storage.csv",
            "data/processed_rl_core/torrentialRain_S1_storm_12h_predrain_plant.csv",
            "data/processed_rl_core/torrentialRain_S0_storm_pre_realistic.csv",
        ]
    else:
        candidates = [
            "data/processed_rl_core/moderateRain_M1_medium_water_level.csv",
            "data/processed/moderateRain_M1_medium_water_level.csv",
            "data/processed_rl_core/moderateRain_M0_medium_pre.csv",
            "data/processed_rl_core/moderateRain_M2_medium_fsn_storage.csv",
        ]

    for rel in candidates:
        path = ROOT_DIR / rel
        if path.exists():
            return str(path)
    return None


def _parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _safe_action(action: Any, action_dim: int = 3) -> np.ndarray:
    arr = np.asarray(action).reshape(-1)
    if arr.size != action_dim:
        arr = np.zeros(action_dim, dtype=np.float32)
    return np.clip(np.rint(arr), 0, 1).astype(np.float32)


@dataclass
class EvalResult:
    policy: str
    start_idx: int
    steps: int
    total_overflow: float
    max_water: float
    avg_risk: float
    cumulative_reward: float
    total_energy: float
    terminated: bool
    truncated: bool


def run_episode(
    policy: str,
    start_idx: int,
    csv_path: str | None,
    model,
    use_normalized_obs: bool,
    max_steps: int,
) -> EvalResult:
    env = DrainageEnv(swmm_csv_path=csv_path, random_start=False, max_steps=max_steps)
    obs, _ = env.reset(options={"start_idx": start_idx})

    total_overflow = 0.0
    total_risk = 0.0
    cumulative_reward = 0.0
    total_energy = 0.0
    max_water = float(np.max(obs[1:4]))
    terminated = False
    truncated = False
    steps = 0

    for step in range(max_steps):
        if policy == "ppo":
            model_obs = env.normalize_observation(obs) if use_normalized_obs else obs
            action, _ = model.predict(model_obs, deterministic=True)
            action = _safe_action(action, action_dim=env.action_dim)
        elif policy == "all_off":
            action = np.zeros(env.action_dim, dtype=np.float32)
        elif policy == "all_on":
            action = np.ones(env.action_dim, dtype=np.float32)
        elif policy == "random":
            action = env.action_space.sample().astype(np.float32)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_overflow += float(info.get("overflow", 0.0))
        total_risk += float(info.get("risk", 0.0))
        cumulative_reward += float(reward)
        total_energy += float(info.get("energy", 0.0))
        max_water = max(max_water, float(np.max(obs[1:4])))
        steps = step + 1

        if terminated or truncated:
            break

    avg_risk = total_risk / steps if steps else 0.0
    return EvalResult(
        policy=policy,
        start_idx=start_idx,
        steps=steps,
        total_overflow=total_overflow,
        max_water=max_water,
        avg_risk=avg_risk,
        cumulative_reward=cumulative_reward,
        total_energy=total_energy,
        terminated=terminated,
        truncated=truncated,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed-start drainage scenarios")
    default_model_path, default_meta_path = _resolve_default_model_paths()
    parser.add_argument(
        "--model-path",
        default=os.getenv("EVAL_MODEL_PATH", "").strip() or str(default_model_path),
        help="PPO model .zip path. Defaults to TRAIN_MODE-matched saved model.",
    )
    parser.add_argument(
        "--meta-path",
        default=os.getenv("EVAL_META_PATH", "").strip() or str(default_meta_path),
        help="Model metadata .json path. Defaults to TRAIN_MODE-matched saved metadata.",
    )
    parser.add_argument(
        "--csv-path",
        default=os.getenv("EVAL_CSV_PATH", "").strip(),
        help="Evaluation CSV path. Defaults to model-matched CSV from metadata.",
    )
    parser.add_argument(
        "--starts",
        default=os.getenv("EVAL_STARTS", "0,7,10"),
        help="Comma-separated start indices, e.g. 0,7,10",
    )
    parser.add_argument(
        "--policies",
        default=os.getenv("EVAL_POLICIES", "ppo,all_off,all_on,random"),
        help="Comma-separated policies: ppo,all_off,all_on,random",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("EVAL_MAX_STEPS", "50")),
        help="Rollout horizon per start index.",
    )
    parser.add_argument(
        "--output-csv",
        default=os.getenv("EVAL_OUTPUT_CSV", "").strip(),
        help="Optional path to save results as CSV.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    meta_path = Path(args.meta_path)
    csv_path = args.csv_path or _resolve_default_csv(meta_path)
    if csv_path is None:
        print("未找到可用于评估的 CSV。请手动传 --csv-path。")
        return 1

    starts = _parse_int_list(args.starts)
    policies = [p.strip().lower() for p in args.policies.split(",") if p.strip()]

    meta = _load_model_meta(meta_path)
    use_normalized_obs = bool(meta.get("obs_normalized", False))

    model = None
    if "ppo" in policies:
        if PPO is None:
            print("stable-baselines3 不可用，无法评估 PPO。")
            return 1
        if not model_path.exists():
            print(f"未找到模型文件: {model_path}")
            return 1
        model = PPO.load(str(model_path))

    print(f"评估模型: {model_path}")
    print(f"模型元数据: {meta_path}")
    print(f"评估 CSV: {csv_path}")
    print(f"起点: {starts}")
    print(f"策略: {policies}")
    print(f"观测是否归一化: {use_normalized_obs}")
    print("-" * 88)

    results: list[EvalResult] = []
    for start_idx in starts:
        for policy in policies:
            result = run_episode(
                policy=policy,
                start_idx=start_idx,
                csv_path=csv_path,
                model=model,
                use_normalized_obs=use_normalized_obs,
                max_steps=args.max_steps,
            )
            results.append(result)
            print(
                f"{policy:8s} | start={start_idx:3d} | steps={result.steps:3d} | "
                f"overflow={result.total_overflow:8.2f} | max_water={result.max_water:5.2f} | "
                f"avg_risk={result.avg_risk:7.2f} | reward={result.cumulative_reward:8.2f} | "
                f"energy={result.total_energy:6.2f}"
            )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "policy",
                    "start_idx",
                    "steps",
                    "total_overflow",
                    "max_water",
                    "avg_risk",
                    "cumulative_reward",
                    "total_energy",
                    "terminated",
                    "truncated",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.policy,
                        r.start_idx,
                        r.steps,
                        round(r.total_overflow, 6),
                        round(r.max_water, 6),
                        round(r.avg_risk, 6),
                        round(r.cumulative_reward, 6),
                        round(r.total_energy, 6),
                        int(r.terminated),
                        int(r.truncated),
                    ]
                )
        print(f"结果已保存到: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
