from __future__ import annotations

import argparse
from pathlib import Path

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


def compress_one(src: Path, dst: Path, keep_time: bool = True) -> tuple[int, int]:
    df = pd.read_csv(src)
    keep = keep_rl_core_columns(list(df.columns), keep_time=keep_time)
    out = df[keep].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(df.columns), len(out.columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed_rl_core")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--keep-time", action="store_true", default=True)
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
        before, after = compress_one(src, dst, keep_time=args.keep_time)
        print(f"[ok] {src.name}: cols {before} -> {after}")
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()

