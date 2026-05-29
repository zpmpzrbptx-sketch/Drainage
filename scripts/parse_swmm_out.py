from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_COLS = [
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


def extract_with_swmm_api(out_path: Path) -> pd.DataFrame:
    from swmm_api import read_out_file  # type: ignore

    out = read_out_file(str(out_path))

    # 兼容 swmm_api 版本差异：优先从 labels 读取对象名
    labels = getattr(out, "labels", {}) or {}
    node_names: List[str] = list(labels.get("node", []))
    link_names: List[str] = list(labels.get("link", []))
    st_names: List[str] = list(labels.get("subcatchment", []))

    if len(node_names) < 3 or len(link_names) < 3 or len(st_names) < 3:
        raise ValueError(
            f"Object count too small in {out_path.name}: "
            f"nodes={len(node_names)}, links={len(link_names)}, subcatchments={len(st_names)}"
        )

    times = pd.to_datetime(out.index)

    rain_series = out.get_part("subcatchment", st_names[0], "rainfall")

    data: Dict[str, pd.Series] = {
        "time": times,
        "rain": pd.Series(rain_series, index=times).astype("float32"),
    }

    for i, node in enumerate(node_names[:3], start=1):
        s = out.get_part("node", node, "depth")
        data[f"water_{i}"] = pd.Series(s, index=times).astype("float32")

    for i, link in enumerate(link_names[:3], start=1):
        s = out.get_part("link", link, "flow")
        data[f"flow_{i}"] = pd.Series(s, index=times).astype("float32")

    for i, st in enumerate(st_names[:3], start=1):
        # 这里用 runoff 作为 storage 代理特征，先保证训练流程可跑；
        # 后续可按你论文定义改为更精确的蓄积指标。
        s = out.get_part("subcatchment", st, "runoff")
        data[f"storage_{i}"] = pd.Series(s, index=times).astype("float32")

    df = pd.DataFrame(data)
    df = df.dropna().reset_index(drop=True)

    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"missing required col: {c}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swmm-root", default="data/swmm", help="Root directory of SWMM out/rpt files")
    parser.add_argument("--output-dir", default="data/processed", help="CSV output dir")
    args = parser.parse_args()

    swmm_root = Path(args.swmm_root)
    source_map = {
        "base_8": swmm_root / "8.out",
        "arid_S1": swmm_root / "arid" / "S1_station_network.out",
        "arid_S2": swmm_root / "arid" / "S2_fsn_capacity_flush.out",
        "moderateRain_S1": swmm_root / "moderateRain" / "S1_water_level_priority.out",
        "moderateRain_S2": swmm_root / "moderateRain" / "S2_balanced_energy_quality.out",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, path in source_map.items():
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue

        print(f"[parse] {name} <- {path}")
        df = extract_with_swmm_api(path)

        # 训练环境只要求数值列，这里额外保留 time 方便分析
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[ok] {csv_path} rows={len(df)}")

    print("done")


if __name__ == "__main__":
    main()
