from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _keep_columns(cols: list[str]) -> list[str]:
    keep: list[str] = []
    for c in cols:
        if c in ("system|rainfall", "system|lateral_inflow"):
            keep.append(c)
            continue
        if c.startswith("node|") and (c.endswith("|depth") or c.endswith("|volume")):
            keep.append(c)
            continue
        if c.startswith("link|") and c.endswith("|flow"):
            keep.append(c)
            continue
    # 保留原顺序并去重
    seen = set()
    ordered = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def slim_one(src: Path, dst: Path) -> tuple[int, int]:
    df = pd.read_csv(src)
    keep = _keep_columns(list(df.columns))
    out = df[keep].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(df.columns), len(out.columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed_slim")
    parser.add_argument("--pattern", default="*.csv")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"[skip] no csv found in {in_dir} with pattern {args.pattern}")
        return

    print(f"[start] slimming {len(files)} files")
    for src in files:
        dst = out_dir / src.name
        before, after = slim_one(src, dst)
        print(f"[ok] {src.name}: cols {before} -> {after}")
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()

