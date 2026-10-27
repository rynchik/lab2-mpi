"""Замеры двух задач внутри одного MPI-запуска."""

import json
from pathlib import Path
import sys
import time

from mpi4py import MPI
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import matrix_vector, vector  # noqa: E402
from src.dot_product import parallel_dot  # noqa: E402
from src.transpose_matvec import parallel_transpose  # noqa: E402


def main():
    comm = MPI.COMM_WORLD
    a = A = x = None
    if comm.rank == 0:
        rng = np.random.default_rng(42)
        a = np.arange(1, 1000001, dtype=np.float64)
        A, x = rng.random((1000, 500)), rng.random(1000)
    rows = []
    cases = [
        (
            "dot_reduce",
            lambda: parallel_dot(a, "reduce", comm),
            lambda: np.dot(vector(a), a),
        ),
        (
            "dot_allreduce",
            lambda: parallel_dot(a, "allreduce", comm),
            lambda: np.dot(vector(a), a),
        ),
        (
            "transpose",
            lambda: parallel_transpose(A, x, comm),
            lambda: np.dot(matrix_vector(A, x)[0].T, x),
        ),
    ]
    for task, parallel, sequential in cases:
        reference = None
        row = {"task": task, "processes": comm.size}
        if comm.rank == 0:
            seq_samples = []
            for trial in range(35):
                start = time.perf_counter()
                reference = sequential()
                elapsed = time.perf_counter() - start
                if trial >= 5:
                    seq_samples.append(elapsed)
            row["sequential_samples"] = seq_samples
            row["sequential_s"] = float(np.median(seq_samples))
        samples, absolute_errors, relative_errors = [], [], []
        for trial in range(35):
            comm.Barrier()
            start = MPI.Wtime()
            result = parallel()
            elapsed = MPI.Wtime() - start
            elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)
            if comm.rank == 0:
                np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)
                difference = np.atleast_1d(result) - np.atleast_1d(reference)
                absolute = float(np.max(np.abs(difference)))
                relative = float(
                    np.linalg.norm(difference)
                    / max(np.linalg.norm(np.atleast_1d(reference)), 1e-300)
                )
                if trial >= 5:
                    samples.append(elapsed)
                    absolute_errors.append(absolute)
                    relative_errors.append(relative)
        if comm.rank == 0:
            row.update(
                parallel_samples=samples,
                parallel_s=float(np.median(samples)),
                max_absolute_error=max(absolute_errors),
                max_relative_error=max(relative_errors),
            )
            rows.append(row)
    if comm.rank == 0:
        print(json.dumps({"mpi": MPI.Get_library_version().strip(), "rows": rows}))


if __name__ == "__main__":
    main()
