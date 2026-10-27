"""Проверки внутри настоящего MPI-запуска; ошибки ранга завершают весь запуск."""

from pathlib import Path
import sys

from mpi4py import MPI
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dot_product import parallel_dot  # noqa: E402
from src.transpose_matvec import parallel_transpose  # noqa: E402


def check():
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(123)
    for M, N in [(1, 1), (2, 3), (7, 5), (8, 4), (31, 17)]:
        A = rng.normal(size=(M, N))
        x = rng.normal(size=M)
        for mode in ("reduce", "allreduce"):
            result = parallel_dot(x if comm.rank == 0 else None, mode, comm)
            if mode == "allreduce" or comm.rank == 0:
                np.testing.assert_allclose(result, x @ x, rtol=1e-12, atol=1e-12)
            else:
                assert result is None
        result = parallel_transpose(
            A if comm.rank == 0 else None, x if comm.rank == 0 else None, comm
        )
        if comm.rank == 0:
            np.testing.assert_allclose(result, A.T @ x, rtol=1e-12, atol=1e-12)
        else:
            assert result is None
    # Ручной пример: [1,2,3] @ [1,2,3] = 14.
    result = parallel_dot(
        np.arange(1.0, 4.0) if comm.rank == 0 else None, "allreduce", comm
    )
    assert result == 14.0
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = np.array([1.0, 2.0, 3.0])
    result = parallel_transpose(
        A if comm.rank == 0 else None, x if comm.rank == 0 else None, comm
    )
    if comm.rank == 0:
        np.testing.assert_array_equal(result, [22.0, 28.0])
    for invalid in ("empty", "dimensions", "mode"):
        caught = False
        try:
            if invalid == "empty":
                parallel_dot(np.array([]) if comm.rank == 0 else None, comm=comm)
            elif invalid == "dimensions":
                parallel_transpose(
                    A if comm.rank == 0 else None,
                    np.ones(2) if comm.rank == 0 else None,
                    comm,
                )
            else:
                parallel_dot(x if comm.rank == 0 else None, mode="wrong", comm=comm)
        except ValueError:
            caught = True
        assert comm.allreduce(int(caught), op=MPI.SUM) == comm.size
    if comm.rank == 0:
        print(f"MPI cases OK: {comm.size} processes")


if __name__ == "__main__":
    try:
        check()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
