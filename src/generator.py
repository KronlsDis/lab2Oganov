from numpy import random, savetxt
from pathlib import Path

# Размеры матрицы
M = 7
N = 4

# Путь к папке data (на уровень выше src)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Создание папки data, если её ещё нет
DATA_DIR.mkdir(exist_ok=True)

# Генерация матрицы A размером M x N
A = random.randint(1, 10, size=(M, N)).astype(float)

# Генерация вектора x длины M
x = random.randint(1, 10, size=M).astype(float)

# Запись размеров матрицы в in.dat
# Сначала N, затем M (как в лекции)
with open(DATA_DIR / "in.dat", "w") as f:
    f.write(f"{N}\n")
    f.write(f"{M}\n")

# Запись матрицы построчно
savetxt(DATA_DIR / "AData.dat", A.reshape(-1), fmt="%.1f")

# Запись вектора
savetxt(DATA_DIR / "xData.dat", x, fmt="%.1f")

print("Файлы успешно созданы!")
print(f"Размер матрицы: {M} x {N}")
print(f"Длина вектора: {M}")
print("Папка:", DATA_DIR)

print("\nМатрица A:")
print(A)

print("\nВектор x:")
print(x)
