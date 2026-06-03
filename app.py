import os
import json
import re
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, cast

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import pymysql
except Exception:  # pragma: no cover - 兼容未安装 MySQL 客户端驱动的环境
    pymysql = None

from env.drainage_env import DrainageEnv

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - 兼容未安装 SB3 的环境
    PPO = None

def _resolve_secret_key() -> str:
    env_key = os.getenv("FLASK_SECRET_KEY", "").strip()
    if env_key:
        return env_key
    key_path = os.path.join(os.path.dirname(__file__), ".flask_secret_key")
    try:
        if os.path.exists(key_path):
            with open(key_path, "r", encoding="utf-8") as f:
                persisted = f.read().strip()
            if persisted:
                return persisted
        generated = os.urandom(32).hex()
        with open(key_path, "w", encoding="utf-8") as f:
            f.write(generated)
        return generated
    except Exception:
        return os.urandom(32).hex()


app = Flask(__name__)
app.secret_key = _resolve_secret_key()
logger = logging.getLogger("drainage")


MODEL_PATH = "models/ppo_drainage.zip"
MODEL_META_PATH = "models/ppo_drainage.meta.json"


def _resolve_default_swmm_csv() -> str:
    # 未显式配置 SWMM_CSV_PATH 时，默认切到真实场景 CSV（优先中雨场景）
    train_mode = ""
    if os.path.exists(MODEL_META_PATH):
        try:
            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            train_mode = str(meta.get("train_mode", "")).strip().lower()
        except Exception:
            train_mode = ""

    if train_mode.startswith("single_m0") or train_mode.startswith("multi_m0"):
        candidates = [
            "data/processed_rl_core/moderateRain_M0_medium_pre.csv",
            "data/processed_rl_core/moderateRain_M1_medium_water_level.csv",
            "data/processed_rl_core/moderateRain_M2_medium_fsn_storage.csv",
            "data/processed/moderateRain_M0_medium_pre.csv",
            "data/processed/moderateRain_M1_medium_water_level.csv",
            "data/processed/moderateRain_M2_medium_fsn_storage.csv",
        ]
    else:
        candidates = [
            "data/processed_rl_core/moderateRain_M1_medium_water_level.csv",
            "data/processed_rl_core/moderateRain_M0_medium_pre.csv",
            "data/processed_rl_core/moderateRain_M2_medium_fsn_storage.csv",
            "data/processed/moderateRain_M1_medium_water_level.csv",
            "data/processed/moderateRain_M0_medium_pre.csv",
            "data/processed/moderateRain_M2_medium_fsn_storage.csv",
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


SWMM_CSV_PATH = os.getenv("SWMM_CSV_PATH", "").strip() or _resolve_default_swmm_csv()
MONITOR_SESSION_ID = "monitor-dashboard"
DECISION_STRATEGY_VERSION = "rule-v1.2.0"
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1").strip()
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root").strip()
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "").strip()
MYSQL_DB = os.getenv("MYSQL_DB", "drainage").strip()
MYSQL_CHARSET = os.getenv("MYSQL_CHARSET", "utf8mb4").strip()
ADMIN_INIT_TOKEN = os.getenv("ADMIN_INIT_TOKEN", "").strip()


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


ALLOW_PUBLIC_REGISTER = _env_flag("ALLOW_PUBLIC_REGISTER", default=False)
FLASK_DEBUG = _env_flag("FLASK_DEBUG", default=False)


def _db_not_ready() -> tuple[Any, int]:
    return (
        jsonify(
            {
                "error": "mysql_not_ready",
                "message": "MySQL 驱动或连接不可用，请先安装 PyMySQL 并配置 MYSQL_* 环境变量。",
            }
        ),
        500,
    )


def get_db_connection():
    if pymysql is None:
        return None
    try:
        return pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset=MYSQL_CHARSET,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )
    except Exception as e:
        logger.exception("MySQL connection failed: %s", e)
        return None


def _err(error: str, message: str, status: int = 500, exc: Exception | None = None):
    if exc is not None:
        logger.exception("%s: %s", error, exc)
    return jsonify({"error": error, "message": message}), status


def _normalize_datetime(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("T", " ")
    try:
        dt = datetime.fromisoformat(text)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d":
                return dt.strftime("%Y-%m-%d 00:00:00")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
    return None


def _safe_float_or_none(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _safe_int(raw: Any, default: int = 0, min_v: int | None = None, max_v: int | None = None) -> int:
    try:
        val = int(raw)
    except Exception:
        val = int(default)
    if min_v is not None:
        val = max(min_v, val)
    if max_v is not None:
        val = min(max_v, val)
    return val


def _current_user() -> dict[str, Any] | None:
    uid = session.get("uid")
    username = session.get("username")
    role = session.get("role")
    if not uid or not username or not role:
        return None
    return {"id": uid, "username": username, "role": role}


def _valid_username(username: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", username or ""))


def require_login(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if _current_user() is None:
            if request.path.startswith("/api/"):
                return jsonify({"error": "unauthorized", "message": "请先登录。"}), 401
            return redirect(url_for("admin_login_page"))
        return fn(*args, **kwargs)

    return wrapper


def require_roles(*roles: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = _current_user()
            if user is None:
                return jsonify({"error": "unauthorized", "message": "请先登录。"}), 401
            if str(user["role"]) not in set(roles):
                return jsonify({"error": "forbidden", "message": "当前角色无权限执行该操作。"}), 403
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def init_admin_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_users (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(64) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('admin','user') NOT NULL DEFAULT 'user',
                is_active TINYINT(1) NOT NULL DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory_data (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                time DATETIME NOT NULL,
                rain DOUBLE DEFAULT NULL,
                water_1 DOUBLE DEFAULT NULL,
                water_2 DOUBLE DEFAULT NULL,
                water_3 DOUBLE DEFAULT NULL,
                flow_1 DOUBLE DEFAULT NULL,
                flow_2 DOUBLE DEFAULT NULL,
                flow_3 DOUBLE DEFAULT NULL,
                storage_1 DOUBLE DEFAULT NULL,
                storage_2 DOUBLE DEFAULT NULL,
                storage_3 DOUBLE DEFAULT NULL,
                inflow DOUBLE DEFAULT NULL,
                created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_time (time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
    conn.commit()


def ensure_default_admin(conn, username: str, password: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, username, role FROM admin_users WHERE username=%s LIMIT 1", (username,))
        row = cur.fetchone()
        if row:
            return {"created": False, "user": row}
        pwd_hash = generate_password_hash(password)
        cur.execute(
            "INSERT INTO admin_users (username, password_hash, role, is_active) VALUES (%s, %s, 'admin', 1)",
            (username, pwd_hash),
        )
        uid = cur.lastrowid
    conn.commit()
    return {"created": True, "user": {"id": uid, "username": username, "role": "admin"}}


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
        self.model_metadata = self._load_rl_model_metadata()
        self.use_normalized_obs = bool(self.model_metadata.get("obs_normalized", False))
        self.rl_model = self._load_rl_model()

    def _load_rl_model_metadata(self) -> dict[str, Any]:
        if not os.path.exists(MODEL_META_PATH):
            return {}
        try:
            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta if isinstance(meta, dict) else {}
        except Exception:
            return {}

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

    def _normalize_model_obs(self, session: SimSession) -> np.ndarray | None:
        if session.obs is None:
            return None
        if not self.use_normalized_obs:
            return session.obs
        env = session.env
        if env is not None and hasattr(env, "normalize_observation"):
            try:
                drainage_env = cast(DrainageEnv, env)
                return drainage_env.normalize_observation(session.obs)
            except Exception:
                pass
        return session.obs

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
                model_obs = self._normalize_model_obs(session)
                if model_obs is None:
                    return _safe_action([0] * action_dim, action_dim=action_dim, pump_mask=pump_mask), "none"
                model_action, _ = self.rl_model.predict(model_obs, deterministic=True)
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


def _jsonify_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.strftime("%Y-%m-%d %H:%M:%S")
        else:
            out[k] = v
    return out


def _derive_risk_level(overflow: float, water_levels: list[float]) -> str:
    max_water = max(water_levels) if water_levels else 0.0
    if overflow > 0.0 or max_water >= 8.0:
        return "high"
    if max_water >= 6.0:
        return "medium"
    return "low"


def _parse_record_payload(payload: dict[str, Any], partial: bool = False) -> tuple[dict[str, Any] | None, str | None]:
    scenario = str(payload.get("scenario", "default")).strip() or "default"
    scenario = scenario[:64]

    record_time = _normalize_datetime(payload.get("record_time"))
    if not partial and record_time is None:
        return None, "record_time 格式无效或缺失，示例：2026-05-30 15:20:00"

    metric_defaults = {
        "rain": 0.0,
        "water_node1": 0.0,
        "water_node2": 0.0,
        "water_node3": 0.0,
        "flow_node1": 0.0,
        "flow_node2": 0.0,
        "flow_node3": 0.0,
        "storage_node1": 0.0,
        "storage_node2": 0.0,
        "storage_node3": 0.0,
        "inflow": 0.0,
        "energy": 0.0,
        "overflow": 0.0,
        "reward": 0.0,
    }

    metrics: dict[str, float] = {}
    for key, default in metric_defaults.items():
        val = _safe_float_or_none(payload.get(key))
        if val is None:
            if partial:
                continue
            metrics[key] = float(default)
        else:
            metrics[key] = float(val)

    if partial and record_time is None and not metrics and "scenario" not in payload and "mode" not in payload and "remark" not in payload and "risk_level" not in payload:
        return None, "至少提供一个需要更新的字段。"

    mode = str(payload.get("mode", "rule")).strip() or "rule"
    mode = mode[:32]
    remark_raw = payload.get("remark")
    remark = None if remark_raw is None else str(remark_raw).strip()[:255]

    risk_level = str(payload.get("risk_level", "")).strip().lower()
    if risk_level not in {"low", "medium", "high"}:
        water_levels = [
            float(metrics.get("water_node1", _safe_float_or_none(payload.get("water_node1")) or 0.0)),
            float(metrics.get("water_node2", _safe_float_or_none(payload.get("water_node2")) or 0.0)),
            float(metrics.get("water_node3", _safe_float_or_none(payload.get("water_node3")) or 0.0)),
        ]
        overflow = float(metrics.get("overflow", _safe_float_or_none(payload.get("overflow")) or 0.0))
        risk_level = _derive_risk_level(overflow, water_levels)

    record: dict[str, Any] = {"scenario": scenario, "mode": mode, "risk_level": risk_level}
    if record_time is not None:
        record["record_time"] = record_time
    if remark is not None:
        record["remark"] = remark
    for k, v in metrics.items():
        record[k] = float(v)
    return record, None


def _collect_record_filters(query_args, include_mode: bool = True) -> tuple[list[str], list[Any]]:
    filters: list[str] = []
    params: list[Any] = []

    scenario = str(query_args.get("scenario", "")).strip()
    if scenario:
        filters.append("scenario = %s")
        params.append(scenario)

    if include_mode:
        mode = str(query_args.get("mode", "")).strip()
        if mode:
            filters.append("mode = %s")
            params.append(mode)

    risk_level = str(query_args.get("risk_level", "")).strip().lower()
    if risk_level in {"low", "medium", "high"}:
        filters.append("risk_level = %s")
        params.append(risk_level)

    start_time = _normalize_datetime(query_args.get("start_time"))
    if start_time:
        filters.append("record_time >= %s")
        params.append(start_time)

    end_time = _normalize_datetime(query_args.get("end_time"))
    if end_time:
        filters.append("record_time <= %s")
        params.append(end_time)

    min_rain = _safe_float_or_none(query_args.get("min_rain"))
    if min_rain is not None:
        filters.append("rain >= %s")
        params.append(min_rain)

    max_rain = _safe_float_or_none(query_args.get("max_rain"))
    if max_rain is not None:
        filters.append("rain <= %s")
        params.append(max_rain)

    min_overflow = _safe_float_or_none(query_args.get("min_overflow"))
    if min_overflow is not None:
        filters.append("overflow >= %s")
        params.append(min_overflow)

    max_overflow = _safe_float_or_none(query_args.get("max_overflow"))
    if max_overflow is not None:
        filters.append("overflow <= %s")
        params.append(max_overflow)

    keyword = str(query_args.get("keyword", "")).strip()
    if keyword:
        filters.append("(scenario LIKE %s OR remark LIKE %s)")
        like_kw = f"%{keyword}%"
        params.extend([like_kw, like_kw])

    min_water = _safe_float_or_none(query_args.get("min_water"))
    if min_water is not None:
        filters.append("GREATEST(water_node1, water_node2, water_node3) >= %s")
        params.append(min_water)

    max_water = _safe_float_or_none(query_args.get("max_water"))
    if max_water is not None:
        filters.append("GREATEST(water_node1, water_node2, water_node3) <= %s")
        params.append(max_water)

    return filters, params


def _inventory_sql_exprs() -> dict[str, str]:
    max_water = "GREATEST(COALESCE(water_1, 0), COALESCE(water_2, 0), COALESCE(water_3, 0))"
    overflow = (
        "GREATEST(COALESCE(water_1, 0) - 8.0, 0) + "
        "GREATEST(COALESCE(water_2, 0) - 8.0, 0) + "
        "GREATEST(COALESCE(water_3, 0) - 8.0, 0)"
    )
    risk = (
        f"CASE WHEN ({overflow}) > 0 OR ({max_water}) >= 8.0 THEN 'high' "
        f"WHEN ({max_water}) >= 6.0 THEN 'medium' ELSE 'low' END"
    )
    avg_water = "(COALESCE(water_1, 0) + COALESCE(water_2, 0) + COALESCE(water_3, 0)) / 3.0"
    return {
        "max_water": max_water,
        "overflow": overflow,
        "risk": risk,
        "avg_water": avg_water,
    }


def _collect_inventory_filters(query_args, include_mode: bool = True) -> tuple[list[str], list[Any]]:
    filters: list[str] = []
    params: list[Any] = []
    expr = _inventory_sql_exprs()

    scenario = str(query_args.get("scenario", "")).strip().lower()
    if scenario and scenario not in {"inventory", "default"}:
        filters.append("1=0")

    if include_mode:
        mode = str(query_args.get("mode", "")).strip().lower()
        if mode and mode != "rule":
            filters.append("1=0")

    risk_level = str(query_args.get("risk_level", "")).strip().lower()
    if risk_level in {"low", "medium", "high"}:
        filters.append(f"({expr['risk']}) = %s")
        params.append(risk_level)

    start_time = _normalize_datetime(query_args.get("start_time"))
    if start_time:
        filters.append("time >= %s")
        params.append(start_time)

    end_time = _normalize_datetime(query_args.get("end_time"))
    if end_time:
        filters.append("time <= %s")
        params.append(end_time)

    min_rain = _safe_float_or_none(query_args.get("min_rain"))
    if min_rain is not None:
        filters.append("COALESCE(rain, 0) >= %s")
        params.append(min_rain)

    max_rain = _safe_float_or_none(query_args.get("max_rain"))
    if max_rain is not None:
        filters.append("COALESCE(rain, 0) <= %s")
        params.append(max_rain)

    min_inflow = _safe_float_or_none(query_args.get("min_inflow"))
    if min_inflow is not None:
        filters.append("COALESCE(inflow, 0) >= %s")
        params.append(min_inflow)

    max_inflow = _safe_float_or_none(query_args.get("max_inflow"))
    if max_inflow is not None:
        filters.append("COALESCE(inflow, 0) <= %s")
        params.append(max_inflow)

    min_overflow = _safe_float_or_none(query_args.get("min_overflow"))
    if min_overflow is not None:
        filters.append(f"({expr['overflow']}) >= %s")
        params.append(min_overflow)

    max_overflow = _safe_float_or_none(query_args.get("max_overflow"))
    if max_overflow is not None:
        filters.append(f"({expr['overflow']}) <= %s")
        params.append(max_overflow)

    keyword = str(query_args.get("keyword", "")).strip()
    if keyword:
        like_kw = f"%{keyword}%"
        filters.append("(CAST(id AS CHAR) LIKE %s OR DATE_FORMAT(time, '%%Y-%%m-%%d %%H:%%i:%%s') LIKE %s)")
        params.extend([like_kw, like_kw])

    min_water = _safe_float_or_none(query_args.get("min_water"))
    if min_water is not None:
        filters.append(f"({expr['max_water']}) >= %s")
        params.append(min_water)

    max_water = _safe_float_or_none(query_args.get("max_water"))
    if max_water is not None:
        filters.append(f"({expr['max_water']}) <= %s")
        params.append(max_water)

    return filters, params


@app.route("/admin")
@require_login
def admin_page():
    user = _current_user()
    return render_template("admin.html", logged_in=bool(user), current_user=user)


@app.route("/admin/login")
def admin_login_page():
    user = _current_user()
    if user is not None:
        return redirect(url_for("home"))
    return render_template("login.html", allow_public_register=ALLOW_PUBLIC_REGISTER)


@app.route("/admin/register")
def admin_register_page():
    if not ALLOW_PUBLIC_REGISTER:
        return redirect(url_for("admin_login_page"))
    user = _current_user()
    if user is not None:
        return redirect(url_for("home"))
    return render_template("register.html", allow_public_register=ALLOW_PUBLIC_REGISTER)


# =========================
# 后台管理接口
# =========================


@app.route("/api/admin/init", methods=["POST"])
def admin_init():
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()

    payload = request.get_json(silent=True) or {}
    if ADMIN_INIT_TOKEN:
        req_token = str(payload.get("init_token", "")).strip()
        if req_token != ADMIN_INIT_TOKEN:
            return jsonify({"error": "forbidden", "message": "初始化令牌无效。"}), 403
    bootstrap_user = str(payload.get("username", os.getenv("ADMIN_BOOTSTRAP_USER", "admin"))).strip() or "admin"
    bootstrap_pass = str(payload.get("password", os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "admin123456"))).strip() or "admin123456"

    try:
        init_admin_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(1) AS cnt FROM admin_users")
            count_row = cur.fetchone() or {"cnt": 0}
            user_count = int(count_row.get("cnt", 0))

        user = _current_user()
        if user_count > 0 and (user is None or str(user.get("role")) != "admin"):
            return jsonify({"error": "forbidden", "message": "系统已初始化，仅管理员可再次执行。"}), 403

        created_info = ensure_default_admin(conn, bootstrap_user, bootstrap_pass)
        return jsonify(
            {
                "ok": True,
                "created": created_info["created"],
                "user": created_info["user"],
                "message": "初始化完成。首次部署后请尽快修改默认密码。",
            }
        )
    except Exception as e:
        conn.rollback()
        return _err("init_failed", "初始化失败，请检查日志。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", "")).strip()
    if not username or not password:
        return jsonify({"error": "invalid_input", "message": "请输入用户名和密码。"}), 400

    try:
        init_admin_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash, role, is_active FROM admin_users WHERE username=%s LIMIT 1",
                (username,),
            )
            row = cur.fetchone()
        if row is None or not row.get("is_active"):
            return jsonify({"error": "login_failed", "message": "账号不存在或已禁用。"}), 401
        if not check_password_hash(str(row.get("password_hash")), password):
            return jsonify({"error": "login_failed", "message": "用户名或密码错误。"}), 401
        session["uid"] = int(row["id"])
        session["username"] = str(row["username"])
        session["role"] = str(row["role"])
        return jsonify({"ok": True, "user": {"id": row["id"], "username": row["username"], "role": row["role"]}})
    except Exception as e:
        return _err("login_failed", "登录失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/register", methods=["POST"])
def admin_register():
    if not ALLOW_PUBLIC_REGISTER:
        return jsonify({"error": "forbidden", "message": "当前环境未开放公开注册，请联系管理员创建账号。"}), 403
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", "")).strip()
    role = "user"

    if not username or len(username) > 64:
        return jsonify({"error": "invalid_input", "message": "用户名不能为空且长度不能超过64。"}), 400
    if not _valid_username(username):
        return jsonify({"error": "invalid_input", "message": "用户名仅允许字母/数字/下划线。"}), 400
    if len(password) < 6:
        return jsonify({"error": "invalid_input", "message": "密码至少 6 位。"}), 400

    try:
        init_admin_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM admin_users WHERE username=%s LIMIT 1", (username,))
            if cur.fetchone() is not None:
                return jsonify({"error": "conflict", "message": "用户名已存在。"}), 409
            cur.execute(
                """
                INSERT INTO admin_users (username, password_hash, role, is_active)
                VALUES (%s, %s, %s, 1)
                """,
                (username, generate_password_hash(password), role),
            )
            uid = cur.lastrowid
        conn.commit()
        return jsonify({"ok": True, "id": uid, "message": "注册成功，请登录。"})
    except Exception as e:
        conn.rollback()
        return _err("register_failed", "注册失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/admin/me")
def admin_me():
    user = _current_user()
    if user is None:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "user": user})


@app.route("/api/admin/users")
@require_roles("admin")
def admin_users_list():
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, role, is_active, created_at, updated_at FROM admin_users ORDER BY id DESC"
            )
            rows = [_jsonify_row(x) for x in cur.fetchall()]
        return jsonify({"items": rows, "total": len(rows)})
    except Exception as e:
        return _err("query_failed", "用户查询失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/users", methods=["POST"])
@require_roles("admin")
def admin_users_create():
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", "")).strip()
    role = str(payload.get("role", "user")).strip().lower()
    is_active = 1 if bool(payload.get("is_active", True)) else 0

    if not username or len(username) > 64:
        return jsonify({"error": "invalid_input", "message": "用户名不能为空且长度不能超过64。"}), 400
    if len(password) < 6:
        return jsonify({"error": "invalid_input", "message": "密码至少 6 位。"}), 400
    if role not in {"admin", "user"}:
        return jsonify({"error": "invalid_input", "message": "角色仅支持 admin 或 user。"}), 400

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM admin_users WHERE username=%s LIMIT 1", (username,))
            if cur.fetchone() is not None:
                return jsonify({"error": "conflict", "message": "用户名已存在。"}), 409
            cur.execute(
                """
                INSERT INTO admin_users (username, password_hash, role, is_active)
                VALUES (%s, %s, %s, %s)
                """,
                (username, generate_password_hash(password), role, is_active),
            )
            uid = cur.lastrowid
        conn.commit()
        return jsonify({"ok": True, "id": uid})
    except Exception as e:
        conn.rollback()
        return _err("create_failed", "创建用户失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/users/<int:user_id>", methods=["PUT"])
@require_roles("admin")
def admin_users_update(user_id: int):
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    payload = request.get_json(silent=True) or {}
    sets: list[str] = []
    params: list[Any] = []

    if "role" in payload:
        role = str(payload.get("role", "")).strip().lower()
        if role not in {"admin", "user"}:
            return jsonify({"error": "invalid_input", "message": "角色仅支持 admin 或 user。"}), 400
        sets.append("role=%s")
        params.append(role)
    if "is_active" in payload:
        sets.append("is_active=%s")
        params.append(1 if bool(payload.get("is_active")) else 0)
    if "password" in payload:
        password = str(payload.get("password", "")).strip()
        if len(password) < 6:
            return jsonify({"error": "invalid_input", "message": "密码至少 6 位。"}), 400
        sets.append("password_hash=%s")
        params.append(generate_password_hash(password))

    if not sets:
        return jsonify({"error": "invalid_input", "message": "没有可更新字段。"}), 400

    current = _current_user()
    if current is not None and int(current["id"]) == user_id and "is_active" in payload and not bool(payload.get("is_active")):
        return jsonify({"error": "invalid_operation", "message": "不能禁用当前登录账号。"}), 400

    sql = f"UPDATE admin_users SET {', '.join(sets)} WHERE id=%s"
    params.append(user_id)

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM admin_users WHERE id=%s LIMIT 1", (user_id,))
            if cur.fetchone() is None:
                return jsonify({"error": "not_found", "message": "用户不存在。"}), 404
            cur.execute(sql, tuple(params))
        conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return _err("update_failed", "更新用户失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@require_roles("admin")
def admin_users_delete(user_id: int):
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    current = _current_user()
    if current is not None and int(current["id"]) == user_id:
        return jsonify({"error": "invalid_operation", "message": "不能删除当前登录账号。"}), 400
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM admin_users WHERE id=%s", (user_id,))
            affected = cur.rowcount
        conn.commit()
        if affected == 0:
            return jsonify({"error": "not_found", "message": "用户不存在。"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return _err("delete_failed", "删除用户失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/records", methods=["GET"])
@require_login
def admin_records_list():
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()

    page = _safe_int(request.args.get("page", 1), default=1, min_v=1)
    page_size = _safe_int(request.args.get("page_size", 20), default=20, min_v=1, max_v=200)
    offset = (page - 1) * page_size

    filters, params = _collect_inventory_filters(request.args, include_mode=True)
    expr = _inventory_sql_exprs()

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

    sort_map = {
        "id": "id",
        "record_time": "time",
        "rain": "rain",
        "energy": "0",
        "overflow": expr["overflow"],
        "reward": "0",
        "created_at": "created_at",
        "updated_at": "created_at",
    }
    sort_by_key = str(request.args.get("sort_by", "record_time")).strip().lower()
    sort_by = sort_map.get(sort_by_key, "time")
    sort_order = str(request.args.get("sort_order", "desc")).strip().lower()
    sort_order = "ASC" if sort_order == "asc" else "DESC"

    sql_count = f"SELECT COUNT(1) AS cnt FROM inventory_data {where_sql}"
    sql_data = (
        f"SELECT id, 'inventory' AS scenario, time AS record_time, COALESCE(rain, 0) AS rain, "
        "COALESCE(water_1, 0) AS water_node1, COALESCE(water_2, 0) AS water_node2, COALESCE(water_3, 0) AS water_node3, "
        "COALESCE(flow_1, 0) AS flow_node1, COALESCE(flow_2, 0) AS flow_node2, COALESCE(flow_3, 0) AS flow_node3, "
        "COALESCE(storage_1, 0) AS storage_node1, COALESCE(storage_2, 0) AS storage_node2, COALESCE(storage_3, 0) AS storage_node3, "
        "COALESCE(inflow, 0) AS inflow, "
        f"0 AS energy, ({expr['overflow']}) AS overflow, 0 AS reward, ({expr['risk']}) AS risk_level, "
        "'rule' AS mode, '' AS remark, NULL AS created_by, created_at, created_at AS updated_at "
        f"FROM inventory_data {where_sql} ORDER BY {sort_by} {sort_order} LIMIT %s OFFSET %s"
    )

    try:
        with conn.cursor() as cur:
            cur.execute(sql_count, tuple(params))
            total = int((cur.fetchone() or {}).get("cnt", 0))
            data_params = list(params) + [page_size, offset]
            cur.execute(sql_data, tuple(data_params))
            rows = [_jsonify_row(x) for x in cur.fetchall()]
        return jsonify({"items": rows, "total": total, "page": page, "page_size": page_size})
    except Exception as e:
        return _err("query_failed", "记录查询失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/records/<int:record_id>", methods=["GET"])
@require_login
def admin_records_get(record_id: int):
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    expr = _inventory_sql_exprs()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, 'inventory' AS scenario, time AS record_time, COALESCE(rain, 0) AS rain,
                       COALESCE(water_1, 0) AS water_node1, COALESCE(water_2, 0) AS water_node2, COALESCE(water_3, 0) AS water_node3,
                       COALESCE(flow_1, 0) AS flow_node1, COALESCE(flow_2, 0) AS flow_node2, COALESCE(flow_3, 0) AS flow_node3,
                       COALESCE(storage_1, 0) AS storage_node1, COALESCE(storage_2, 0) AS storage_node2, COALESCE(storage_3, 0) AS storage_node3,
                       COALESCE(inflow, 0) AS inflow,
                       0 AS energy, ({expr['overflow']}) AS overflow, 0 AS reward, ({expr['risk']}) AS risk_level,
                       'rule' AS mode, '' AS remark, NULL AS created_by, created_at, created_at AS updated_at
                FROM inventory_data
                WHERE id=%s
                LIMIT 1
                """,
                (record_id,),
            )
            row = cur.fetchone()
        if row is None:
            return jsonify({"error": "not_found", "message": "记录不存在。"}), 404
        return jsonify({"item": _jsonify_row(row)})
    except Exception as e:
        return _err("query_failed", "记录详情查询失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/records", methods=["POST"])
@require_roles("admin")
def admin_records_create():
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    payload = request.get_json(silent=True) or {}
    record, err = _parse_record_payload(payload, partial=False)
    if err or record is None:
        return jsonify({"error": "invalid_input", "message": err or "数据格式错误。"}), 400

    sql = """
        INSERT INTO inventory_data (
            time, rain, water_1, water_2, water_3,
            flow_1, flow_2, flow_3, storage_1, storage_2, storage_3, inflow
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        record["record_time"],
        float(record.get("rain", 0.0)),
        float(record.get("water_node1", 0.0)),
        float(record.get("water_node2", 0.0)),
        float(record.get("water_node3", 0.0)),
        float(record.get("flow_node1", 0.0)),
        float(record.get("flow_node2", 0.0)),
        float(record.get("flow_node3", 0.0)),
        float(record.get("storage_node1", 0.0)),
        float(record.get("storage_node2", 0.0)),
        float(record.get("storage_node3", 0.0)),
        float(record.get("inflow", 0.0)),
    )

    try:
        with conn.cursor() as cur:
            cur.execute(sql, values)
            rid = cur.lastrowid
        conn.commit()
        return jsonify({"ok": True, "id": rid})
    except Exception as e:
        conn.rollback()
        return _err("create_failed", "新增记录失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/records/<int:record_id>", methods=["PUT"])
@require_roles("admin")
def admin_records_update(record_id: int):
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    payload = request.get_json(silent=True) or {}
    record, err = _parse_record_payload(payload, partial=True)
    if err or record is None:
        return jsonify({"error": "invalid_input", "message": err or "数据格式错误。"}), 400

    sets: list[str] = []
    params: list[Any] = []
    field_map = {
        "record_time": "time",
        "rain": "rain",
        "water_node1": "water_1",
        "water_node2": "water_2",
        "water_node3": "water_3",
        "flow_node1": "flow_1",
        "flow_node2": "flow_2",
        "flow_node3": "flow_3",
        "storage_node1": "storage_1",
        "storage_node2": "storage_2",
        "storage_node3": "storage_3",
    }
    for key, col in field_map.items():
        if key in record:
            sets.append(f"{col}=%s")
            params.append(record[key])
    if "inflow" in payload:
        sets.append("inflow=%s")
        params.append(_safe_float_or_none(payload.get("inflow")))
    if not sets:
        return jsonify({"error": "invalid_input", "message": "没有可更新字段。"}), 400
    params.append(record_id)
    sql = f"UPDATE inventory_data SET {', '.join(sets)} WHERE id=%s"

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM inventory_data WHERE id=%s LIMIT 1", (record_id,))
            if cur.fetchone() is None:
                return jsonify({"error": "not_found", "message": "记录不存在。"}), 404
            cur.execute(sql, tuple(params))
        conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return _err("update_failed", "更新记录失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/records/<int:record_id>", methods=["DELETE"])
@require_roles("admin")
def admin_records_delete(record_id: int):
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM inventory_data WHERE id=%s", (record_id,))
            affected = cur.rowcount
        conn.commit()
        if affected == 0:
            return jsonify({"error": "not_found", "message": "记录不存在。"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        conn.rollback()
        return _err("delete_failed", "删除记录失败，请稍后重试。", 500, e)
    finally:
        conn.close()


@app.route("/api/admin/summary")
@require_login
def admin_summary():
    conn = get_db_connection()
    if conn is None:
        return _db_not_ready()
    group_by = str(request.args.get("group_by", "hour")).strip().lower()
    limit = _safe_int(request.args.get("limit", 72), default=72, min_v=10, max_v=300)
    expr = _inventory_sql_exprs()

    bucket_expr = "DATE_FORMAT(time, '%%Y-%%m-%%d %%H:00:00')"
    if group_by == "day":
        bucket_expr = "DATE_FORMAT(time, '%%Y-%%m-%%d 00:00:00')"

    filters, params = _collect_inventory_filters(request.args, include_mode=True)
    compare_filters, compare_params = _collect_inventory_filters(request.args, include_mode=False)
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    compare_condition_sql = " AND ".join(compare_filters + ["'rule' IN ('rl', 'rule')"])
    compare_where_sql = f"WHERE {compare_condition_sql}" if compare_condition_sql else ""

    sql_series = f"""
        SELECT {bucket_expr} AS bucket,
               COUNT(1) AS count,
               AVG(COALESCE(rain, 0)) AS avg_rain,
               AVG({expr["avg_water"]}) AS avg_water,
               AVG(COALESCE(inflow, 0)) AS avg_inflow,
               0 AS total_energy,
               SUM({expr["overflow"]}) AS total_overflow,
               0 AS avg_reward
        FROM inventory_data
        {where_sql}
        GROUP BY bucket
        ORDER BY bucket DESC
        LIMIT %s
    """
    sql_risk = f"""
        SELECT ({expr["risk"]}) AS risk_level, COUNT(1) AS count
        FROM inventory_data
        {where_sql}
        GROUP BY risk_level
    """
    sql_mode = f"""
        SELECT CASE
                   WHEN COALESCE(inflow, 0) < 10 THEN '0-10'
                   WHEN COALESCE(inflow, 0) < 30 THEN '10-30'
                   WHEN COALESCE(inflow, 0) < 60 THEN '30-60'
                   ELSE '60+'
               END AS `range`,
               COUNT(1) AS count
        FROM inventory_data
        {where_sql}
        GROUP BY `range`
        ORDER BY MIN(COALESCE(inflow, 0))
    """
    sql_kpi = f"""
        SELECT COUNT(1) AS total,
               AVG(COALESCE(rain, 0)) AS avg_rain,
               AVG({expr["avg_water"]}) AS avg_water,
               AVG(COALESCE(inflow, 0)) AS avg_inflow,
               MAX({expr["max_water"]}) AS peak_water,
               0 AS total_energy,
               SUM({expr["overflow"]}) AS total_overflow,
               0 AS avg_reward
        FROM inventory_data
        {where_sql}
    """
    sql_rl_vs_rule = f"""
        SELECT 'rule' AS mode,
               COUNT(1) AS count,
               AVG({expr["avg_water"]}) AS avg_water,
               0 AS total_energy,
               SUM({expr["overflow"]}) AS total_overflow,
               0 AS avg_reward
        FROM inventory_data
        {compare_where_sql}
    """

    try:
        with conn.cursor() as cur:
            cur.execute(sql_series, tuple(params + [limit]))
            rows = [_jsonify_row(x) for x in cur.fetchall()]
            rows.reverse()

            cur.execute(sql_risk, tuple(params))
            risk_rows = [_jsonify_row(x) for x in cur.fetchall()]

            cur.execute(sql_mode, tuple(params))
            mode_rows = [_jsonify_row(x) for x in cur.fetchall()]

            cur.execute(sql_kpi, tuple(params))
            kpi = _jsonify_row(cur.fetchone() or {})

            cur.execute(sql_rl_vs_rule, tuple(compare_params))
            rl_rule_rows = [_jsonify_row(x) for x in cur.fetchall()]

        rl_vs_rule = {
            "rule": {
                "count": 0,
                "avg_water": 0.0,
                "total_energy": 0.0,
                "total_overflow": 0.0,
                "avg_reward": 0.0,
            },
            "rl": {
                "count": 0,
                "avg_water": 0.0,
                "total_energy": 0.0,
                "total_overflow": 0.0,
                "avg_reward": 0.0,
            },
        }
        for row in rl_rule_rows:
            mode = str(row.get("mode", "")).strip().lower()
            if mode not in {"rule", "rl"}:
                continue
            rl_vs_rule[mode] = {
                "count": int(row.get("count") or 0),
                "avg_water": round(float(row.get("avg_water") or 0.0), 3),
                "total_energy": round(float(row.get("total_energy") or 0.0), 3),
                "total_overflow": round(float(row.get("total_overflow") or 0.0), 3),
                "avg_reward": round(float(row.get("avg_reward") or 0.0), 3),
            }

        compare_by_mode = str(request.args.get("mode", "")).strip() == ""

        return jsonify(
            {
                "series": rows,
                "risk_distribution": risk_rows,
                "mode_distribution": mode_rows,
                "kpi": {
                    "total": int(kpi.get("total") or 0),
                    "avg_rain": round(float(kpi.get("avg_rain") or 0.0), 3),
                    "avg_water": round(float(kpi.get("avg_water") or 0.0), 3),
                    "avg_inflow": round(float(kpi.get("avg_inflow") or 0.0), 3),
                    "peak_water": round(float(kpi.get("peak_water") or 0.0), 3),
                    "total_energy": round(float(kpi.get("total_energy") or 0.0), 3),
                    "total_overflow": round(float(kpi.get("total_overflow") or 0.0), 3),
                    "avg_reward": round(float(kpi.get("avg_reward") or 0.0), 3),
                },
                "rl_vs_rule": rl_vs_rule,
                "rl_vs_rule_enabled": compare_by_mode,
            }
        )
    except Exception as e:
        return _err("query_failed", "统计查询失败，请稍后重试。", 500, e)
    finally:
        conn.close()

# =========================
# 页面路由
# =========================


# 首页
@app.route("/", endpoint="home")
def home():
    user = _current_user()
    if user is None:
        return redirect(url_for("admin_login_page"))
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

    app.run(debug=FLASK_DEBUG)
