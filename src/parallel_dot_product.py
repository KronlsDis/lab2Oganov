from time import perf_counter

from mpi4py import MPI
from numpy import arange, empty, int32, float64, dot, array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

if rank == 0:
    start_time = perf_counter()
    M = 20
    a = arange(1, M + 1, dtype=float64)
    
    ave, res = divmod(M, numprocs - 1)

    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    
    rcounts [0] = 0
    displs [0] = 0

    for k in range(1, numprocs):
        if k <= res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
   
        displs[k] = displs[k - 1] + rcounts[k - 1]

else:
    a = None
    rcounts = None
    displs = None

M_part = array(0, dtype=int32)
comm.Scatter([rcounts, MPI.INT], [M_part, MPI.INT], root=0)

a_part = empty(M_part, dtype=float64)
comm.Scatterv([a, rcounts, displs, MPI.DOUBLE], [a_part, MPI.DOUBLE], root=0)

ScalP_temp = empty(1, dtype=float64)
ScalP_temp[0] = dot(a_part, a_part)

ScalP = empty(1, dtype=float64)

comm.Reduce([ScalP_temp, MPI.DOUBLE], [ScalP, MPI.DOUBLE], op=MPI.SUM, root=0)

if rank == 0:
    elapsed_time = perf_counter() - start_time
    sequential_start = perf_counter()
    ScalP_seq = dot(a, a)
    sequential_elapsed = perf_counter() - sequential_start

    print(f"Параллельный результат: {ScalP[0]:.1f}")
    print(f"Последовательный результат: {ScalP_seq:.1f}")

    if abs(ScalP[0] - ScalP_seq) < 1e-10:
        print("Проверка пройдена: результаты совпадают.")
    else:
        print("Ошибка: результаты не совпадают.")

    print(f"Время выполнения: {elapsed_time:.9f} секунд")
    print(f"Время последовательного выполнения: {sequential_elapsed:.9f} секунд")
