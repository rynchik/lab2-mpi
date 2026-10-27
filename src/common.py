"""Разбиение данных и согласованная проверка входа на всех рангах."""

from pathlib import Path

import numpy as np


def partition(M, size):
    """Число строк и начальная строка каждого процесса."""
    if M < 0 or size <= 0:
        raise ValueError("Требуются M >= 0 и size > 0")
    quotient, remainder = divmod(M, size)
    counts = np.full(size, quotient, dtype=np.int32)
    counts[:remainder] += 1
    offsets = np.zeros(size, dtype=np.int32)
    offsets[1:] = np.cumsum(counts[:-1])
    return counts, offsets


def root_input(comm, load):
    """Вызвать load только на корне; сообщить об ошибке всем процессам."""
    value = None
    error = None
    if comm.rank == 0:
        try:
            value = load()
        except (OSError, ValueError, TypeError) as exc:
            error = str(exc)
    error = comm.bcast(error, root=0)
    if error is not None:
        raise ValueError(error)
    return value


def vector(value):
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("Требуется непустой одномерный вектор")
    if not np.isfinite(result).all():
        raise ValueError("Элементы должны быть конечными числами")
    return result


def matrix_vector(A, x):
    A = np.ascontiguousarray(A, dtype=np.float64)
    x = vector(x)
    if A.ndim != 2 or min(A.shape) <= 0 or A.shape[0] != x.size:
        raise ValueError("Требуются A размера M x N и x длины M")
    if not np.isfinite(A).all():
        raise ValueError("Элементы матрицы должны быть конечными числами")
    return A, x


def read_data(directory):
    """in.dat содержит N, затем M, как в первой лабораторной."""
    directory = Path(directory)
    tokens = (directory / "in.dat").read_text(encoding="utf-8").split()
    if len(tokens) != 2:
        raise ValueError("in.dat: сначала N, затем M, ровно два целых числа")
    N, M = map(int, tokens)
    if min(M, N) <= 0:
        raise ValueError("Размеры матрицы должны быть положительными")
    flat = np.loadtxt(directory / "AData.dat", dtype=np.float64).reshape(-1)
    if flat.size != M * N:
        raise ValueError("Количество элементов матрицы не равно M * N")
    x = np.loadtxt(directory / "xData.dat", dtype=np.float64).reshape(-1)
    return matrix_vector(flat.reshape(M, N), x)
