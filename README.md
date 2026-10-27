# Лабораторная работа №2. Коллективные операции MPI

Вычисление скалярного произведения `a @ a` через `Reduce` и `Allreduce`, а также произведения `A.T @ x` через `Scatterv` и `Reduce`.

Отчёт, результаты измерений и объяснение алгоритмов находятся в [report.md](report.md).

## Запуск

Нужны Python и реализация MPI. В Linux можно использовать Open MPI или MPICH. В Windows для локальной проверки использован Intel MPI из пакета `impi-rt`.

```bash
python -m pip install -r requirements.txt
mpiexec -n 4 python src/dot_product.py --size 1000000 --mode reduce
mpiexec -n 4 python src/dot_product.py --size 1000000 --mode allreduce
mpiexec -n 4 python src/transpose_matvec.py
```

В `data/` находятся файлы для матрицы `1000 × 500`. Порядок чисел в `in.dat`: сначала число столбцов `N`, затем число строк `M`. Длина вектора `x` равна `M`.

Набор с неравномерным разбиением (`M=7`, `N=5`):

```bash
mpiexec -n 4 python src/transpose_matvec.py --data-dir data/uneven
```

Программы сами сравнивают результат с NumPy. Вариант `Allreduce` дополнительно выводит результат на каждом ранге.

## Проверки и измерения

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/benchmark.py
```

`mpiexec` должен быть в `PATH`; для тестов и скрипта измерений можно вместо этого задать переменную `MPIEXEC` с полным путём к нему. В Windows MPI можно установить в виртуальное окружение командой `python -m pip install impi-rt`.

Скрипт `benchmark.py` обновляет `report/performance.csv`, `report/raw_timings.json` и `report/speedup.png`. После повторного эксперимента нужно обновить числа и выводы в отчёте.
