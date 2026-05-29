import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

from env.drainage_env import DrainageEnv

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - 兼容未安装 SB3 的环境
    PPO = None

app = Flask(__name__)


MODEL_PATH = "models/ppo_drainage.zip"
SWMM_CSV_PATH = os.getenv("SWMM_CSV_PATH", "").strip()
MONITOR_SESSION_ID = "monitor-dashboard"
DECISION_STRATEGY_VERSION = "rule-v1.2.0"


def _safe_float(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float32).reshape(-1)[0])


def _safe_action(
    action: Any,
    action_dim: int = 3,
    pump_mask: Any = None,
) -> list[int]:
    action_dim = int(max(1, action_dim))
    arr = np.asarray(action).reshape(-1)
    arr = np.clip(np.rint(arr), 0, 1).astype(np.int32)
    if arr.size < action_dim:
        arr = np.pad(arr, (0, action_dim - arr.size), mode="constant")
    arr = arr[:action_dim]
    if pump_mask is not None:
        mask = np.asarray(pump_mask).reshape(-1)
        if mask.size < action_dim:
            mask = np.pad(mask, (0, action_dim - mask.size), mode="constant")
        mask = np.clip(np.rint(mask[:action_dim]), 0, 1).astype(np.int32)
        arr = arr * mask
    return arr.tolist()


def _risk_level(overflow: float, max_water: float) -> str:
    if overflow > 0.0 or max_water >= 8.0:
        return "high"
    if max_water >= 6.0:
        return "medium"
    return "low"


def _rule_action(
    obs: np.ndarray,
    action_dim: int = 3,
    pump_mask: Any = None,
) -> list[int]:
    rain = float(obs[0])
    w1, w2, w3 = [float(x) for x in obs[1:4]]
    action = [0] * int(max(1, action_dim))
    if w1 > 6.0 or rain > 8.0:
        action[0] = 1
    if action_dim >= 2 and (w2 > 6.0 or rain > 8.0):
        action[1] = 1
    if action_dim >= 3 and (w3 > 6.0 or rain > 8.0):
        action[2] = 1
    if max(w1, w2, w3) > 7.5:
        action = [1] * int(max(1, action_dim))
    return _safe_action(action, action_dim=action_dim, pump_mask=pump_mask)


def _manual_or_zero(
    action: Any,
    action_dim: int = 3,
    pump_mask: Any = None,
) -> list[int]:
    if action is None:
        return _safe_action([0] * int(max(1, action_dim)), action_dim=action_dim, pump_mask=pump_mask)
    return _safe_action(action, action_dim=action_dim, pump_mask=pump_mask)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _monitor_events(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for snap in trace[-8:]:
        step = int(snap.get("step", 0))
        rain = float(snap.get("rain", 0.0))
        overflow = float(snap.get("overflow", 0.0))
        water_levels = [float(x) for x in snap.get("water_levels", [0.0, 0.0, 0.0])]
        max_water = max(water_levels) if water_levels else 0.0
        pump_on = int(sum(int(v) for v in snap.get("action", [0, 0, 0])))

        if overflow > 0.0:
            level = "danger"
            message = f"Step {step}: 发生溢流 {overflow:.2f} m³，建议提高排水负荷。"
        elif max_water >= 6.5:
            level = "warn"
            message = f"Step {step}: 峰值水位 {max_water:.2f} m，接近高风险阈值。"
        elif rain > 8.0 and pump_on == 0:
            level = "warn"
            message = f"Step {step}: 强降雨 {rain:.2f} mm/h，建议提前开泵预排。"
        else:
            level = "safe"
            message = f"Step {step}: 系统运行平稳，当前开启泵站数 {pump_on}。"

        events.append(
            {
                "time": f"Step {step}",
                "level": level,
                "message": message,
            }
        )
    events.reverse()
    return events


@dataclass
class SimSession:
    session_id: str
    mode: str = "rule"
    max_steps: int = 200
    seed: int | None = None
    env: DrainageEnv | None = None
    obs: np.ndarray | None = None
    done: bool = False
    step_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.env = DrainageEnv(
            swmm_csv_path=SWMM_CSV_PATH or None,
            max_steps=self.max_steps,
        )
        self.obs, _ = self.env.reset(seed=self.seed)
        self.done = False
        self.step_count = 0
        self.history = []


class SimulationService:
    def __init__(self) -> None:
        self.sessions: dict[str, SimSession] = {}
        self.rl_model = self._load_rl_model()

    def _load_rl_model(self):
        if PPO is None:
            return None
        if not os.path.exists(MODEL_PATH):
            return None
        try:
            return PPO.load(MODEL_PATH)
        except Exception:
            return None

    def create_session(
        self,
        mode: str = "rule",
        max_steps: int = 200,
        seed: int | None = None,
        session_id: str | None = None,
        register: bool = True,
    ) -> SimSession:
        sid = session_id or str(uuid.uuid4())
        session = SimSession(
            session_id=sid,
            mode=mode if mode in {"rl", "rule", "manual"} else "rule",
            max_steps=int(max_steps),
            seed=seed,
        )
        session.reset()
        if register:
            self.sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> SimSession | None:
        return self.sessions.get(session_id)

    def _get_action_spec(self, session: SimSession) -> tuple[int, Any]:
        env = session.env
        if env is None:
            return 3, np.ones(3, dtype=np.int32)
        action_dim = int(getattr(env, "action_dim", 3))
        pump_mask = getattr(env, "pump_mask", np.ones(action_dim, dtype=np.int32))
        return action_dim, pump_mask

    def decide_action(self, session: SimSession, manual_action: Any = None) -> tuple[list[int], str]:
        action_dim, pump_mask = self._get_action_spec(session)
        if session.obs is None:
            return _safe_action([0] * action_dim, action_dim=action_dim, pump_mask=pump_mask), "none"
        if session.mode == "manual":
            action = _manual_or_zero(
                manual_action,
                action_dim=action_dim,
                pump_mask=pump_mask,
            )
            return action, "manual"
        if session.mode == "rl":
            if self.rl_model is not None:
                model_action, _ = self.rl_model.predict(session.obs, deterministic=True)
                return _safe_action(
                    model_action,
                    action_dim=action_dim,
                    pump_mask=pump_mask,
                ), "rl"
            return _rule_action(
                session.obs,
                action_dim=action_dim,
                pump_mask=pump_mask,
            ), "rule_fallback"
        return _rule_action(
            session.obs,
            action_dim=action_dim,
            pump_mask=pump_mask,
        ), "rule"

    def explain(self, snapshot: dict[str, Any]) -> str:
        overflow = float(snapshot["overflow"])
        rain = float(snapshot["rain"])
        max_water = float(max(snapshot["water_levels"]))
        action = snapshot["action"]
        if overflow > 0:
            return "出现溢流，优先提升排水负荷并维持泵站开启。"
        if max_water > 6.5:
            return "水位接近高风险阈值，执行预排以释放管网容量。"
        if rain > 8.0 and sum(action) == 0:
            return "降雨偏强但尚未触发高水位，系统保持观察状态。"
        if rain > 8.0:
            return "降雨增强，采取提前排水避免后续水位抬升。"
        return "系统运行平稳，维持经济性优先的调度策略。"

    def step(self, session: SimSession, manual_action: Any = None) -> dict[str, Any]:
        if session.env is None or session.obs is None:
            session.reset()
        if session.done:
            return {
                "session_id": session.session_id,
                "done": True,
                "step": session.step_count,
                "message": "Simulation ended, please reset.",
            }
        action, action_source = self.decide_action(session, manual_action=manual_action)
        next_obs, reward, terminated, truncated, info = session.env.step(action)  # type: ignore[union-attr]
        session.obs = next_obs
        session.step_count += 1
        session.done = bool(terminated or truncated)

        rain = _safe_float(next_obs[0])
        water_levels = [float(x) for x in next_obs[1:4]]
        flow_levels = [float(x) for x in next_obs[4:7]]
        storage_levels = [float(x) for x in next_obs[7:10]]
        overflow = float(info.get("overflow", 0.0))
        energy = float(info.get("energy", 0.0))
        max_water = float(max(water_levels))

        snapshot = {
            "session_id": session.session_id,
            "mode": session.mode,
            "action_source": action_source,
            "step": session.step_count,
            "done": session.done,
            "rain": rain,
            "water_levels": water_levels,
            "flow_levels": flow_levels,
            "storage_levels": storage_levels,
            "overflow": overflow,
            "energy": energy,
            "reward": float(reward),
            "action": action,
            "risk": _risk_level(overflow, max_water),
            "active_pump_count": int(
                info.get("active_pump_count", getattr(session.env, "active_pump_count", 3))
            ),
            "pump_mask": info.get(
                "pump_mask",
                np.asarray(
                    getattr(session.env, "pump_mask", np.ones(3, dtype=np.int32))
                )
                .astype(np.int32)
                .tolist(),
            ),
        }
        snapshot["explain"] = self.explain(snapshot)
        session.history.append(snapshot)
        return snapshot

    def reset(self, session: SimSession) -> dict[str, Any]:
        session.reset()
        assert session.obs is not None
        action_dim, pump_mask = self._get_action_spec(session)
        zero_action = _safe_action(
            [0] * action_dim,
            action_dim=action_dim,
            pump_mask=pump_mask,
        )
        water_levels = [float(x) for x in session.obs[1:4]]
        flow_levels = [float(x) for x in session.obs[4:7]]
        storage_levels = [float(x) for x in session.obs[7:10]]
        overflow = float(np.sum(np.maximum(np.asarray(water_levels) - 8.0, 0.0)))
        max_water = float(max(water_levels))
        return {
            "session_id": session.session_id,
            "mode": session.mode,
            "step": 0,
            "done": False,
            "rain": _safe_float(session.obs[0]),
            "water_levels": water_levels,
            "flow_levels": flow_levels,
            "storage_levels": storage_levels,
            "overflow": overflow,
            "energy": 0.0,
            "reward": 0.0,
            "action": zero_action,
            "action_source": "reset",
            "risk": _risk_level(overflow, max_water),
            "active_pump_count": int(getattr(session.env, "active_pump_count", action_dim)),
            "pump_mask": np.asarray(pump_mask).astype(np.int32).tolist(),
            "explain": "系统已重置，等待下一步调度。",
        }

    def history(self, session: SimSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "mode": session.mode,
            "steps": len(session.history),
            "history": session.history,
        }

    def compare(self, steps: int = 120, seed: int | None = 42) -> dict[str, Any]:
        steps = int(np.clip(steps, 10, 1000))
        rule_session = self.create_session(
            mode="rule", max_steps=steps + 5, seed=seed, register=False
        )
        rl_session = self.create_session(
            mode="rl", max_steps=steps + 5, seed=seed, register=False
        )

        rule_trace: list[dict[str, Any]] = []
        rl_trace: list[dict[str, Any]] = []
        for _ in range(steps):
            r = self.step(rule_session)
            q = self.step(rl_session)
            rule_trace.append(r)
            rl_trace.append(q)
            if r.get("done") and q.get("done"):
                break

        def agg(trace: list[dict[str, Any]]) -> dict[str, float]:
            if not trace:
                return {"total_overflow": 0.0, "total_energy": 0.0, "avg_reward": 0.0}
            total_overflow = float(sum(float(x["overflow"]) for x in trace))
            total_energy = float(sum(float(x["energy"]) for x in trace))
            avg_reward = float(np.mean([float(x["reward"]) for x in trace]))
            return {
                "total_overflow": round(total_overflow, 3),
                "total_energy": round(total_energy, 3),
                "avg_reward": round(avg_reward, 3),
            }

        return {
            "seed": seed,
            "steps": len(rule_trace),
            "rule_metrics": agg(rule_trace),
            "rl_metrics": agg(rl_trace),
            "rule_trace": rule_trace,
            "rl_trace": rl_trace,
        }


sim_service = SimulationService()

# =========================
# 页面路由
# =========================


# 首页
@app.route("/")
def index():

    return render_template("index.html")


# 数据监控页
@app.route("/charts")
def charts():

    return render_template("charts.html")


# 决策解释页
@app.route("/decision")
def decision():

    return render_template("decision.html")


# 动态仿真页
@app.route("/simulation")
def simulation():

    return render_template("simulation.html")


# =========================
# 图表数据接口
# =========================


@app.route("/api/data")
def get_data():
    steps = int(np.clip(int(request.args.get("steps", 20)), 10, 120))

    monitor_session = sim_service.get_session(MONITOR_SESSION_ID)
    if monitor_session is None:
        monitor_session = sim_service.create_session(
            mode="rule",
            max_steps=max(steps + 10, 80),
            session_id=MONITOR_SESSION_ID,
            seed=42,
        )

    if monitor_session.done:
        sim_service.reset(monitor_session)

    # 持续推进监控会话，保证每次拉取都有新数据
    for _ in range(3):
        if monitor_session.done:
            break
        sim_service.step(monitor_session)

    history = monitor_session.history[-steps:]
    if not history:
        history = [sim_service.reset(monitor_session)]

    rain = [float(x["rain"]) for x in history]
    water = [
        [float(x["water_levels"][0]) for x in history],
        [float(x["water_levels"][1]) for x in history],
        [float(x["water_levels"][2]) for x in history],
    ]
    energy = [float(x["energy"]) for x in history]
    overflow = [float(x["overflow"]) for x in history]
    reward = [float(x["reward"]) for x in history]
    flow = [
        [float(x["flow_levels"][0]) for x in history],
        [float(x["flow_levels"][1]) for x in history],
        [float(x["flow_levels"][2]) for x in history],
    ]
    storage = [
        [float(x["storage_levels"][0]) for x in history],
        [float(x["storage_levels"][1]) for x in history],
        [float(x["storage_levels"][2]) for x in history],
    ]
    action_dim = (
        int(getattr(monitor_session.env, "action_dim", 3))
        if monitor_session.env is not None
        else 3
    )
    actions = [x.get("action", [0] * action_dim) for x in history]

    flat_water = [v for row in water for v in row]
    avg_water_series = [float((water[0][i] + water[1][i] + water[2][i]) / 3.0) for i in range(len(history))]
    risk_score = float(
        np.clip(
            (rain[-1] if rain else 0.0) * 4.0
            + max(0.0, (avg_water_series[-1] if avg_water_series else 0.0) - 4.5) * 20.0
            + min(sum(overflow), 5.0) * 8.0,
            0.0,
            100.0,
        )
    )
    system_score = float(np.clip(100.0 - risk_score * 0.6 - sum(energy) * 0.08, 0.0, 100.0))

    on_counts = [0] * action_dim
    for act in actions:
        safe_act = _safe_action(act, action_dim=action_dim)
        for i in range(action_dim):
            on_counts[i] += int(safe_act[i])
    total_steps = max(len(actions), 1)
    pump_on_ratio = [round(c / total_steps * 100.0, 2) for c in on_counts]

    compare_data = sim_service.compare(steps=min(60, max(20, steps)), seed=42)
    rule_metrics = compare_data["rule_metrics"]
    rl_metrics = compare_data["rl_metrics"]

    events = _monitor_events(history)
    risk_level = "high" if risk_score >= 75 else ("medium" if risk_score >= 45 else "low")

    return jsonify(
        {
            # 兼容旧前端（index/main.js、历史 charts.js）
            "rain": rain,
            "water": water,
            "energy": energy,
            "compare": {
                "rule": [
                    float(rule_metrics["total_overflow"]),
                    float(rule_metrics["total_energy"]),
                    float(rule_metrics["avg_reward"]),
                ],
                "rl": [
                    float(rl_metrics["total_overflow"]),
                    float(rl_metrics["total_energy"]),
                    float(rl_metrics["avg_reward"]),
                ],
            },
            "meta": {
                "session_id": monitor_session.session_id,
                "mode": monitor_session.mode,
                "step": int(history[-1].get("step", 0)) if history else 0,
                "steps": len(history),
                "risk_level": risk_level,
            },
            "series": {
                "rain": rain,
                "water": water,
                "energy": energy,
                "overflow": overflow,
                "reward": reward,
                "flow": flow,
                "storage": storage,
                "actions": actions,
            },
            "kpi": {
                "rain_now": round(rain[-1] if rain else 0.0, 3),
                "avg_water_now": round(avg_water_series[-1] if avg_water_series else 0.0, 3),
                "peak_water": round(max(flat_water) if flat_water else 0.0, 3),
                "total_energy": round(sum(energy), 3),
                "total_overflow": round(sum(overflow), 3),
                "risk_score": round(risk_score, 3),
                "system_score": round(system_score, 3),
                "pump_on_ratio": pump_on_ratio,
                "avg_reward": round(_mean(reward), 3),
                "avg_flow": round(
                    _mean(
                        [
                            (flow[0][i] + flow[1][i] + flow[2][i]) / 3.0
                            for i in range(len(history))
                        ]
                    ),
                    3,
                ),
            },
            "compare": {
                "categories": ["总溢流", "总能耗", "平均奖励"],
                "rule": [
                    float(rule_metrics["total_overflow"]),
                    float(rule_metrics["total_energy"]),
                    float(rule_metrics["avg_reward"]),
                ],
                "rl": [
                    float(rl_metrics["total_overflow"]),
                    float(rl_metrics["total_energy"]),
                    float(rl_metrics["avg_reward"]),
                ],
            },
            "events": events,
        }
    )


# =========================
# 决策解释接口
# =========================


@app.route("/api/decision")
def decision_api():
    water = float(np.random.uniform(0, 10))
    rain = float(np.random.uniform(0, 10))
    overflow = max(0.0, water - 8.0)
    decision_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    decision_id = f"DEC-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    rule_hits = [
        {
            "name": "水位超阈值",
            "expr": "water > 7.0",
            "hit": water > 7.0,
            "weight": 0.55,
            "value": round(water, 2),
            "threshold": 7.0,
            "operator": ">",
        },
        {
            "name": "强降雨预排",
            "expr": "rain > 6.0",
            "hit": rain > 6.0,
            "weight": 0.35,
            "value": round(rain, 2),
            "threshold": 6.0,
            "operator": ">",
        },
        {
            "name": "溢流保护",
            "expr": "overflow > 0",
            "hit": overflow > 0.0,
            "weight": 0.10,
            "value": round(overflow, 2),
            "threshold": 0.0,
            "operator": ">",
        },
    ]

    if overflow > 0.0 or water > 7.0:
        action = "开启泵站（高负荷排水）"
        reason = "水位接近或超过安全阈值，优先控制溢流风险。"
    elif rain > 6.0:
        action = "提前预排（降低水位）"
        reason = "降雨强度较大，需要预留管网容量。"
    else:
        action = "维持当前运行状态"
        reason = "当前水位与降雨均在可控区间。"

    confidence_score = 0.55
    confidence_score += 0.25 if rule_hits[0]["hit"] else 0.0
    confidence_score += 0.15 if rule_hits[1]["hit"] else 0.0
    confidence_score += 0.20 if rule_hits[2]["hit"] else 0.0

    uncertainty: list[str] = []
    if abs(water - 7.0) < 0.25:
        confidence_score -= 0.12
        uncertainty.append("水位接近阈值，边界决策敏感。")
    if abs(rain - 6.0) < 0.25:
        confidence_score -= 0.08
        uncertainty.append("降雨接近阈值，短时波动可能改变策略。")
    if not uncertainty:
        uncertainty.append("主要来自降雨预测误差与传感器采样延迟。")

    confidence_score = float(np.clip(confidence_score, 0.0, 0.99))
    if confidence_score >= 0.8:
        confidence_level = "高"
    elif confidence_score >= 0.6:
        confidence_level = "中"
    else:
        confidence_level = "低"

    def _sensitivity_status(distance: float) -> tuple[str, str]:
        if distance >= 0:
            return "已越界", "建议立即执行保守调度。"
        if distance >= -0.3:
            return "临界区", "建议保持高频观测并准备切换策略。"
        return "安全区", "当前距离阈值充足。"

    sensitivity = []
    for item in rule_hits:
        distance = float(item["value"]) - float(item["threshold"])
        status, advice = _sensitivity_status(distance)
        sensitivity.append(
            {
                "name": item["name"],
                "metric": item["expr"],
                "distance": round(distance, 2),
                "status": status,
                "advice": advice,
            }
        )

    base_risk = float(np.clip(water * 7.2 + rain * 5.8 + overflow * 16.0, 0.0, 100.0))

    def _eval_action(pump_action: list[int], name: str) -> dict[str, Any]:
        pump_on = int(sum(pump_action))
        risk_after = float(
            np.clip(base_risk - pump_on * 14.0 + (8.0 if rain > 8.0 and pump_on == 0 else 0.0), 0.0, 100.0)
        )
        energy_after = float(np.clip(pump_on * 12.0 + max(0.0, rain - 4.0) * 1.5, 0.0, 80.0))
        risk_change = round(risk_after - base_risk, 2)
        return {
            "name": name,
            "action": pump_action,
            "risk_change": risk_change,
            "energy_change": round(energy_after, 2),
            "note": "风险下降明显" if risk_change <= -15 else ("风险可控" if risk_change <= -5 else "风险改善有限"),
        }

    counterfactuals = [
        _eval_action([0, 0, 0], "保持不变"),
        _eval_action([1, 0, 0], "单泵预排"),
        _eval_action([1, 1, 1], "全泵联动"),
    ]
    counterfactuals.sort(key=lambda x: (x["risk_change"], x["energy_change"]))
    if counterfactuals:
        counterfactuals[0]["recommended"] = True

    explain_signature = f"SIG-{int(round((water * 13 + rain * 17 + confidence_score * 100), 0))}-{int(base_risk)}"

    return jsonify(
        {
            "water": round(water, 2),
            "rain": round(rain, 2),
            "action": action,
            "reason": reason,
            "decision_time": decision_time,
            "strategy_version": DECISION_STRATEGY_VERSION,
            "decision_id": decision_id,
            "explain_signature": explain_signature,
            "rule_hits": rule_hits,
            "sensitivity": sensitivity,
            "counterfactuals": counterfactuals,
            "review_window_sec": 120 if confidence_score < 0.6 else 300,
            "confidence": {
                "score": round(confidence_score, 2),
                "level": confidence_level,
                "uncertainty": uncertainty,
            },
        }
    )


# =========================
# 动态仿真接口
# =========================


@app.route("/api/simulation")
def simulation_api():
    # 兼容旧页面：自动创建默认会话并推进一步
    session = sim_service.get_session("legacy-default")
    if session is None:
        session = sim_service.create_session(mode="rule", session_id="legacy-default")
    data = sim_service.step(session)
    water = data["water_levels"]
    flow = data["flow_levels"]
    return jsonify(
        {
            "rain": round(float(data["rain"]), 2),
            "pumpStatus": bool(sum(data["action"]) > 0),
            "node1": round(float(water[0]), 2),
            "node2": round(float(water[1]), 2),
            "flow": round(float(np.mean(flow)), 2),
            "energy": round(float(data["energy"]), 2),
            "overflow": round(float(data["overflow"]), 2),
            "strategy": "强化学习" if data["action_source"] == "rl" else "规则调度",
        }
    )


@app.route("/api/sim/init", methods=["POST"])
def sim_init():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "rule"))
    max_steps = int(payload.get("max_steps", 200))
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else None
    session_id = payload.get("session_id")
    session = sim_service.create_session(
        mode=mode,
        max_steps=max_steps,
        seed=seed,
        session_id=session_id,
    )
    return jsonify(
        {
            "session_id": session.session_id,
            "mode": session.mode,
            "max_steps": session.max_steps,
            "seed": session.seed,
            "done": session.done,
            "step": session.step_count,
        }
    )


@app.route("/api/sim/step", methods=["POST"])
def sim_step():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    manual_action = payload.get("manual_action")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session = sim_service.get_session(str(session_id))
    if session is None:
        return jsonify({"error": "session not found"}), 404
    data = sim_service.step(session, manual_action=manual_action)
    return jsonify(data)


@app.route("/api/sim/reset", methods=["POST"])
def sim_reset():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session = sim_service.get_session(str(session_id))
    if session is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify(sim_service.reset(session))


@app.route("/api/sim/history")
def sim_history():
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session = sim_service.get_session(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify(sim_service.history(session))


@app.route("/api/sim/compare")
def sim_compare():
    steps = int(request.args.get("steps", 120))
    seed = request.args.get("seed")
    seed_int = int(seed) if seed is not None else 42
    return jsonify(sim_service.compare(steps=steps, seed=seed_int))


# =========================
# 启动
# =========================

if __name__ == "__main__":

    app.run(debug=True)
