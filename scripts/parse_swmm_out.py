from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _as_float32_series(values, index: pd.DatetimeIndex) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != len(index):
        raise ValueError(f"series length mismatch: got={arr.size}, expect={len(index)}")
    return pd.Series(arr, index=index).astype("float32")


def _safe_get_part(out, kind: str, name: str, var: str):
    if kind == "system":
        # swmm_api 版本差异：system 有的版本需要占位 name，有的只要 var
        try:
            return out.get_part(kind, name, var)
        except TypeError:
            return out.get_part(kind, var)
    return out.get_part(kind, name, var)


def _escape_col_part(s: str) -> str:
    return str(s).replace("|", "¦")


def _full_extract_with_swmm_api(out_path: Path) -> pd.DataFrame:
    from swmm_api import read_out_file  # type: ignore

    out = read_out_file(str(out_path))

    labels = getattr(out, "labels", {}) or {}
    variables = getattr(out, "variables", {}) or {}

    times = pd.to_datetime(out.index)

    data: Dict[str, pd.Series] = {
        "time": pd.Series(times, index=times),
    }

    for kind in ("system", "subcatchment", "node", "link"):
        var_list = list(variables.get(kind, []))
        if not var_list:
            continue

        if kind == "system":
            for var in var_list:
                values = _safe_get_part(out, "system", "", var)
                col = f"system|{_escape_col_part(var)}"
                data[col] = _as_float32_series(values, times)
            continue

        obj_names: List[str] = list(labels.get(kind, []))
        if not obj_names:
            print(f"[warn:{out_path.name}] no labels for kind={kind}, skipped")
            continue

        for obj in obj_names:
            obj_clean = _escape_col_part(obj)
            for var in var_list:
                values = _safe_get_part(out, kind, obj, var)
                var_clean = _escape_col_part(var)
                col = f"{kind}|{obj_clean}|{var_clean}"
                data[col] = _as_float32_series(values, times)

    df = pd.DataFrame(data)
    return df.reset_index(drop=True)


def _find_out_files(swmm_root: Path) -> List[Path]:
    return sorted([p for p in swmm_root.rglob("*.out") if p.is_file()])


def _csv_name_from_out(swmm_root: Path, out_path: Path) -> str:
    rel = out_path.relative_to(swmm_root)
    stem = rel.with_suffix("")
    safe = "_".join(stem.parts)
    return f"{safe}.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swmm-root", default="data/swmm", help="Root directory of SWMM out files")
    parser.add_argument("--output-dir", default="data/processed", help="CSV output dir")
    args = parser.parse_args()

    swmm_root = Path(args.swmm_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_files = _find_out_files(swmm_root)
    if not out_files:
        print(f"[skip] no .out files found under: {swmm_root}")
        return

    for path in out_files:
        print(f"[parse] {path}")
        df = _full_extract_with_swmm_api(path)

        csv_name = _csv_name_from_out(swmm_root, path)
        csv_path = out_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(f"[ok] {csv_path} rows={len(df)} cols={len(df.columns)}")

    print("done")


if __name__ == "__main__":
    main()
