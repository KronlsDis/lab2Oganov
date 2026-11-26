
from pathlib import Path
from time import perf_counter

from mpi4py import MPI
from numpy import empty, zeros, array, int32, float64, dot, loadtxt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

if rank == 0:
    start_time = perf_counter()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    with (data_dir / "in.dat").open() as dimensions:
        N = int32(int(dimensions.readline()))
        M = int32(int(dimensions.readline()))
else:
    N = array(0, dtype=int32)

comm.Bcast([N, MPI.INT], root=0)

if rank == 0:
    A = loadtxt(data_dir / "AData.dat").reshape(M, N)
    x = loadtxt(data_dir / "xData.dat").reshape(M)

    ave, res = divmod(M, numprocs - 1)

    rcounts = zeros(numprocs, dtype=int32)
    displs = zeros(numprocs, dtype=int32)

    for k in range(1, numprocs):
        if k <= res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave

        if k > 0:
            displs[k] = displs[k - 1] + rcounts[k - 1]

    sendA = rcounts * N
    displsA = displs * N
else:
    A = None
    x = None
    rcounts = None
    displs = None
    sendA = None
    displsA = None

M_part = array(0, dtype=int32)
comm.Scatter([rcounts, MPI.INT], [M_part, MPI.INT], root=0)

A_part = empty((M_part, N), dtype=float64)
comm.Scatterv([A, sendA, displsA, MPI.DOUBLE], [A_part, MPI.DOUBLE], root=0)

x_part = empty(M_part, dtype=float64)
comm.Scatterv([x, rcounts, displs, MPI.DOUBLE], [x_part, MPI.DOUBLE], root=0)

b_temp = dot(A_part.T, x_part)

if rank == 0:
    b = empty(N, dtype=float64)
else:
    b = None

comm.Reduce([b_temp, MPI.DOUBLE], [b, MPI.DOUBLE], op = MPI.SUM, root=0)

if rank == 0:
    elapsed_time = perf_counter() - start_time
    sequential_start = perf_counter()
    b_seq = dot(A.T, x)
    sequential_elapsed = perf_counter() - sequential_start

    print(
        "Параллельный результат: ["
        + " ".join(format(v, ".17g") for v in b)
        + "]"
    )

    print(
        "Последовательный результат: ["
        + " ".join(format(v, ".17g") for v in b_seq)
        + "]"
    )

    if all(abs(b - b_seq) < 1e-10):
        print("Проверка пройдена: результаты совпадают.")
    else:
        print("Ошибка: результаты не совпадают.")

    print(f"Время выполнения: {elapsed_time:.9f} секунд")
    print(f"Время последовательного выполнения: {sequential_elapsed:.9f} секунд")
