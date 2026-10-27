"""Часть 2: b = A.T @ x через два Scatterv и Reduce с MPI.SUM."""

import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np

if __package__:
    from .common import matrix_vector, partition, read_data, root_input
else:
    from common import (  # type: ignore[no-redef]
        matrix_vector,
        partition,
        read_data,
        root_input,
    )


def parallel_transpose(A=None, x=None, comm=MPI.COMM_WORLD):
    """У каждого ранга вклад длины N; итог получает только ранг 0."""
    data = root_input(comm, lambda: matrix_vector(A, x))
    dimensions = np.zeros(2, dtype=np.int64)
    if comm.rank == 0:
        A, x = data
        dimensions[:] = A.shape
    comm.Bcast(dimensions, root=0)
    M, N = map(int, dimensions)
    counts, offsets = partition(M, comm.size)
    local_M = int(counts[comm.rank])
    local_A = np.empty((local_M, N), dtype=np.float64)
    local_x = np.empty(local_M, dtype=np.float64)
    source_A = [A, counts * N, offsets * N, MPI.DOUBLE] if comm.rank == 0 else None
    source_x = [x, counts, offsets, MPI.DOUBLE] if comm.rank == 0 else None
    comm.Scatterv(source_A, local_A, root=0)
    comm.Scatterv(source_x, local_x, root=0)
    local_b = np.dot(local_A.T, local_x)
    result = np.empty(N, dtype=np.float64) if comm.rank == 0 else None
    comm.Reduce(local_b, result, op=MPI.SUM, root=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    try:
        data = root_input(comm, lambda: read_data(args.data_dir))
    except ValueError as exc:
        raise SystemExit(f"Input error: {exc}") from exc
    A, x = data if comm.rank == 0 else (None, None)
    comm.Barrier()
    start = MPI.Wtime()
    result = parallel_transpose(A, x, comm)
    elapsed = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
    if comm.rank == 0:
        expected = np.dot(A.T, x)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        np.savetxt("Results_transpose.dat", result, fmt="%.16e")
        print(f"A={A.shape}, x={x.size}, b={result.size}, processes={comm.size}")
        print("verification=OK")
        print(f"Execution time: {elapsed:.9f} seconds")


if __name__ == "__main__":
    main()
