"""Часть 1: a @ a с распределением Scatterv и суммированием Reduce/Allreduce."""

import argparse

from mpi4py import MPI
import numpy as np

if __package__:
    from .common import partition, root_input, vector
else:
    from common import partition, root_input, vector  # type: ignore[no-redef]


def parallel_dot(a=None, mode="reduce", comm=MPI.COMM_WORLD):
    """Вход задаёт ранг 0; Allreduce возвращает результат каждому рангу."""

    def validate():
        if mode not in ("reduce", "allreduce"):
            raise ValueError("Режим должен быть reduce или allreduce")
        return vector(a)

    a = root_input(comm, validate)
    # Режим выбирает корень: все ранги выполняют одну и ту же редукцию.
    mode = comm.bcast(mode if comm.rank == 0 else None, root=0)
    length = np.array([a.size if comm.rank == 0 else 0], dtype=np.int64)
    comm.Bcast(length, root=0)
    counts, offsets = partition(int(length[0]), comm.size)
    local_a = np.empty(int(counts[comm.rank]), dtype=np.float64)
    source = [a, counts, offsets, MPI.DOUBLE] if comm.rank == 0 else None
    comm.Scatterv(source, local_a, root=0)
    local_dot = np.array([np.dot(local_a, local_a)], dtype=np.float64)
    result = np.empty(1, dtype=np.float64)
    if mode == "allreduce":
        comm.Allreduce(local_dot, result, op=MPI.SUM)
        return float(result[0])
    comm.Reduce(local_dot, result if comm.rank == 0 else None, op=MPI.SUM, root=0)
    return float(result[0]) if comm.rank == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1000000, help="Длина вектора M")
    parser.add_argument("--mode", choices=["reduce", "allreduce"], default="reduce")
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size должен быть положительным")
    comm = MPI.COMM_WORLD
    a = np.arange(1, args.size + 1, dtype=np.float64) if comm.rank == 0 else None
    comm.Barrier()
    start = MPI.Wtime()
    result = parallel_dot(a, args.mode, comm)
    elapsed = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
    if comm.rank == 0:
        expected = float(np.dot(a, a))
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        np.savetxt(f"Results_dot_{args.mode}.dat", [result], fmt="%.16e")
        print(f"mode={args.mode}, M={args.size}, processes={comm.size}")
        print(f"dot={result:.16e}; NumPy={expected:.16e}; verification=OK")
        print(f"Execution time: {elapsed:.9f} seconds")
    if args.mode == "allreduce":
        print(f"rank={comm.rank}: result={result:.16e}", flush=True)


if __name__ == "__main__":
    main()
