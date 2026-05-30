from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def keep_rl_core_columns(cols: list[str], keep_time: bool = True) -> list[str]:
    keep: list[str] = []
    for c in cols:
        if keep_time and c == "time":
            keep.append(c)
            continue
        if c.startswith("node|") and c.endswith("|depth"):
            keep.append(c)
            continue
        if c.startswith("link|") and c.endswith("|flow"):
            keep.append(c)
            continue

    seen: set[str] = set()
    ordered: list[str] = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_top_by_signal(df: pd.DataFrame, candidates: list[str], k: int) -> list[str]:
    if not candidates:
        return []
    scored: list[tuple[float, str]] = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        score = float(s.std()) + 0.1 * float(s.abs().mean())
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def _stack_k(df: pd.DataFrame, selected: list[str], k: int) -> np.ndarray:
    n = len(df)
    if not selected:
        return np.zeros((n, k), dtype=np.float32)

    arrs: list[np.ndarray] = []
    for c in selected:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        arrs.append(s.to_numpy(dtype=np.float32).reshape(-1))

    while len(arrs) < k:
        arrs.append(arrs[-1].copy())
    return np.stack(arrs[:k], axis=1).astype(np.float32)


def _pump_sort_key(col: str) -> tuple[int, str]:
    parts = col.split("|")
    if len(parts) >= 3:
        m = re.fullmatch(r"P(\d+)", parts[1])
        if m:
            return (int(m.group(1)), col)
    return (10**9, col)


FIXED_SCHEMA_COLUMNS = [
    "time",
    "system|rainfall",
    "link|P1|flow",
    "link|P2|flow",
    "link|42722|flow",
    "link|20945|flow",
    "link|21854|flow",
    "link|325|flow",
    "link|12660|flow",
    "link|21823|flow",
    "link|20648|flow",
    "link|20647|flow",
    "link|21822|flow",
    "link|21792|flow",
    "link|21796|flow",
    "link|21788|flow",
    "node|J03260054|depth",
    "node|J03262001|depth",
    "node|J03260051|depth",
    "node|J03262031|depth",
    "node|J03260048|depth",
    "node|J03260041|depth",
    "node|J03261014|depth",
    "node|J03260033|depth",
    "node|J03260038|depth",
    "node|J03261009|depth",
    "node|J03235003|depth",
    "node|J03263052|depth",
    "node|J03260029|depth",
    "node|J03260026|depth",
    "node|J03263039|depth",
    "node|J03411080|depth",
    "node|J03263058|depth",
    "node|J03235002|depth",
    "node|J03190142|depth",
    "node|J03190161|depth",
    "node|J03411020|depth",
    "node|J03190164|depth",
    "node|J03380238|depth",
    "node|J03260559|depth",
    "pump|P1|status",
    "pump|P2|status",
]


def compress_one_shallow(src: Path, dst: Path, keep_time: bool = True) -> tuple[int, int]:
    df = pd.read_csv(src)
    keep = keep_rl_core_columns(list(df.columns), keep_time=keep_time)
    out = df[keep].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(df.columns), len(out.columns)


def compress_one_compact(src: Path, dst: Path, keep_time: bool = True) -> tuple[int, int]:
    df = pd.read_csv(src)
    cols = list(df.columns)

    rain_col = _first_existing(
        df,
        ["system|rainfall", "rain", "subcatchment|S2|rainfall", "subcatchment|S1|rainfall"],
    )
    if rain_col is None:
        rain = np.zeros(len(df), dtype=np.float32)
    else:
        rain = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    depth_cols = [c for c in cols if c.startswith("node|") and c.endswith("|depth")]
    volume_cols = [c for c in cols if c.startswith("node|") and c.endswith("|volume")]
    flow_cols = [c for c in cols if c.startswith("link|") and c.endswith("|flow")]

    top_depth_cols = _pick_top_by_signal(df, depth_cols, 3)
    water = _stack_k(df, top_depth_cols, 3)

    pump_flow_cols = [c for c in flow_cols if re.match(r"^link\|P\d+\|flow$", c)]
    pump_flow_cols = sorted(pump_flow_cols, key=_pump_sort_key)
    top_flow_cols = list(pump_flow_cols[:3])
    if len(top_flow_cols) < 3:
        for c in _pick_top_by_signal(df, flow_cols, 6):
            if c not in top_flow_cols:
                top_flow_cols.append(c)
            if len(top_flow_cols) >= 3:
                break
    if len(top_flow_cols) < 3:
        outflow_col = _first_existing(df, ["system|outflow"])
        if outflow_col is not None:
            top_flow_cols.append(outflow_col)
    flow = _stack_k(df, top_flow_cols, 3)

    top_volume_cols = _pick_top_by_signal(df, volume_cols, 3)
    if top_volume_cols:
        storage = _stack_k(df, top_volume_cols, 3)
    else:
        storage = np.clip(water * 35.0, 0.0, 100.0).astype(np.float32)

    inflow_col = _first_existing(
        df,
        [
            "system|lateral_inflow",
            "system|direct_inflow",
            "system|dry_weather_inflow",
            "system|groundwater_inflow",
            "inflow",
        ],
    )
    if inflow_col is None:
        inflow = np.zeros(len(df), dtype=np.float32)
    else:
        inflow = pd.to_numeric(df[inflow_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    out = pd.DataFrame(
        {
            "rain": rain,
            "water_1": water[:, 0],
            "water_2": water[:, 1],
            "water_3": water[:, 2],
            "flow_1": flow[:, 0],
            "flow_2": flow[:, 1],
            "flow_3": flow[:, 2],
            "storage_1": storage[:, 0],
            "storage_2": storage[:, 1],
            "storage_3": storage[:, 2],
            "inflow": inflow,
        }
    )
    if keep_time and "time" in df.columns:
        out.insert(0, "time", df["time"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(df.columns), len(out.columns)


def compress_one_fixed_schema(src: Path, dst: Path) -> tuple[int, int]:
    df = pd.read_csv(src)
    out = pd.DataFrame(index=df.index)

    for col in FIXED_SCHEMA_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
        elif col == "time":
            out[col] = np.arange(len(df), dtype=np.int64)
        else:
            out[col] = 0.0

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(df.columns), len(out.columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed_rl_core")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--keep-time", action="store_true", default=True)
    parser.add_argument(
        "--mode",
        choices=["compact", "shallow", "fixed"],
        default="compact",
        help="compact: short RL headers; shallow: keep all node depth/link flow; fixed: output the specified legacy schema",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"[skip] no csv found in {in_dir} with pattern {args.pattern}")
        return

    print(f"[start] compress rl-core {len(files)} files")
    for src in files:
        dst = out_dir / src.name
        if args.mode == "compact":
            before, after = compress_one_compact(src, dst, keep_time=args.keep_time)
        elif args.mode == "fixed":
            before, after = compress_one_fixed_schema(src, dst)
        else:
            before, after = compress_one_shallow(src, dst, keep_time=args.keep_time)
        print(f"[ok] {src.name}: cols {before} -> {after}")
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()
