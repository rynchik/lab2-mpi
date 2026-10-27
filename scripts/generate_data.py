"""Создать входные данные A и x для A.T @ x."""

import argparse
from pathlib import Path

import numpy as np


def generate(M, N, seed, output):
    if M <= 0 or N <= 0:
        raise ValueError("M и N должны быть положительными")
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "AData.dat", rng.random((M, N)), fmt="%.16e")
    np.savetxt(output / "xData.dat", rng.random(M), fmt="%.16e")
    (output / "in.dat").write_text(f"{N}\n{M}\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--cols", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    args = parser.parse_args()
    generate(args.rows, args.cols, args.seed, args.output)
