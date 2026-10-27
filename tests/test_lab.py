"""Проверки разбиения, чтения, численных результатов и CLI."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from src.common import partition, read_data

ROOT = Path(__file__).resolve().parents[1]


def run_mpi(processes, script, args=(), cwd=None):
    launcher = os.environ.get("MPIEXEC") or shutil.which("mpiexec")
    assert launcher, "Установите MPI и задайте MPIEXEC или PATH"
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    return subprocess.run(
        [launcher, "-n", str(processes), sys.executable, str(ROOT / script), *args],
        cwd=cwd or ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("M,p", [(0, 4), (1, 4), (7, 4), (8, 4), (2, 8)])
def test_partition(M, p):
    counts, offsets = partition(M, p)
    assert counts.sum() == M
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(counts[:-1])])


def test_reader(tmp_path):
    (tmp_path / "in.dat").write_text("2\n3\n")
    matrix_file = tmp_path / "AData.dat"
    for text in ("1 2\n3 4\n5 6\n", "1\n2\n3\n4\n5\n6\n"):
        matrix_file.write_text(text)
        (tmp_path / "xData.dat").write_text("1\n2\n3\n")
        A, x = read_data(tmp_path)
        np.testing.assert_array_equal(A.T @ x, [22.0, 28.0])
    (tmp_path / "xData.dat").write_text("1\n2\n")
    with pytest.raises(ValueError):
        read_data(tmp_path)
    (tmp_path / "in.dat").write_text("2\n0\n")
    with pytest.raises(ValueError):
        read_data(tmp_path)


@pytest.mark.parametrize("p", [1, 2, 4, 8])
def test_distributed_algorithms(p):
    run = run_mpi(p, "tests/mpi_cases.py")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "MPI cases OK" in run.stdout


@pytest.mark.parametrize("mode", ["reduce", "allreduce"])
def test_dot_cli(mode, tmp_path):
    run = run_mpi(4, "src/dot_product.py", ["--size", "7", "--mode", mode], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert np.loadtxt(tmp_path / f"Results_dot_{mode}.dat") == 140.0
    if mode == "allreduce":
        for rank in range(4):
            assert f"rank={rank}: result=" in run.stdout


def test_transpose_cli(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "in.dat").write_text("2\n3\n")
    np.savetxt(data / "AData.dat", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.savetxt(data / "xData.dat", [1.0, 2.0, 3.0])
    run = run_mpi(4, "src/transpose_matvec.py", ["--data-dir", str(data)], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "Results_transpose.dat"), [22.0, 28.0]
    )


def test_bad_input_no_hang(tmp_path):
    run = run_mpi(4, "src/transpose_matvec.py", ["--data-dir", str(tmp_path)])
    assert run.returncode != 0
    assert "Input error" in run.stdout + run.stderr
