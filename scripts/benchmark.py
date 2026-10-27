"""Измерить время на 1, 2, 4 процессах и построить график по реальным данным."""

import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sequential_baseline(rows, task):
    """Одна общая база для Reduce и Allreduce скалярного произведения."""
    baseline_task = "dot_reduce" if task.startswith("dot_") else task
    return next(
        row["sequential_s"]
        for row in rows
        if row["task"] == baseline_task and row["processes"] == 1
    )


def draw_chart(rows, target):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for task, title, label, ax in [
        ("dot_reduce", "Скалярное произведение", "Reduce", axes[0]),
        ("dot_allreduce", "Скалярное произведение", "Allreduce", axes[0]),
        ("transpose", "Умножение A.T на x", "Scatterv + Reduce", axes[1]),
    ]:
        selected = [row for row in rows if row["task"] == task]
        baseline = sequential_baseline(rows, task)
        ax.plot(
            [row["processes"] for row in selected],
            [baseline / row["parallel_s"] for row in selected],
            "o-",
            label=label,
        )
        ax.set_title(title)
    for ax in axes:
        ax.axhline(1, linestyle="--", color="gray", label="Последовательная версия")
        ax.set_xlabel("Число MPI-процессов")
        ax.set_ylabel("Ускорение относительно NumPy")
        ax.set_xticks([1, 2, 4])
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Коллективные операции MPI; медиана 30 измерений")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    launcher = os.environ.get("MPIEXEC") or shutil.which("mpiexec")
    if not launcher:
        raise SystemExit("Не найден mpiexec: задайте MPIEXEC или PATH")
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    rows, mpi = [], None
    for p in (1, 2, 4):
        result = subprocess.run(
            [
                launcher,
                "-n",
                str(p),
                sys.executable,
                str(ROOT / "scripts/benchmark_worker.py"),
            ],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        output = json.loads(result.stdout.strip().splitlines()[-1])
        rows.extend(output["rows"])
        mpi = output["mpi"]
        print(f"Измерения завершены: {p} процессов", flush=True)
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mpi": mpi,
        "logical_cpus": os.cpu_count(),
        "dot_M": 1000000,
        "matrix_shape": [1000, 500],
        "seed": 42,
        "warmup": 5,
        "repeats": 30,
        "rows": rows,
    }
    save_report(metadata)


def save_report(metadata):
    """Сохранить таблицу и график, в том числе по уже имеющимся замерам."""
    report = ROOT / "report"
    report.mkdir(exist_ok=True)
    rows = metadata["rows"]
    (report / "raw_timings.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    columns = [
        "task",
        "processes",
        "sequential_s",
        "parallel_s",
        "speedup",
        "max_absolute_error",
        "max_relative_error",
    ]
    with (report / "performance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            baseline = sequential_baseline(rows, row["task"])
            saved = {key: row[key] for key in columns if key in row}
            saved.update(sequential_s=baseline, speedup=baseline / row["parallel_s"])
            writer.writerow(saved)
    draw_chart(rows, report / "speedup.png")


if __name__ == "__main__":
    main()
