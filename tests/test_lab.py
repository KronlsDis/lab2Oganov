
import os
import re
import sys
import shutil
import subprocess

from pathlib import Path

import numpy as np
import pytest


# ============================================================
# НАСТРОЙКИ ТЕСТИРОВАНИЯ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"

DOT_SCRIPT = SRC_DIR / "parallel_dot_product.py"
MATVEC_SCRIPT = SRC_DIR / "parallel_matvec.py"

TIMEOUT = 60

# Ищем установленную реализацию MPI
MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")


# ============================================================
# ОБЩИЕ ФУНКЦИИ
# ============================================================

def get_mpi_command(processes, script):
    """
    Формирует команду запуска MPI-программы.
    """

    if MPIEXEC is None:
        pytest.skip("Не найден mpiexec или mpirun. Установите MPI.")

    return [
        MPIEXEC,
        "-n",
        str(processes),
        sys.executable,
        str(script)
    ]


def run_mpi(processes, script, cwd):
    """
    Запускает MPI-программу и возвращает её вывод.
    """

    env = os.environ.copy()

    # Разрешаем запуск OpenMPI от имени root,
    # если это необходимо.
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

    # Разрешаем запуск нескольких процессов,
    # даже если доступно меньше вычислительных слотов.
    env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"

    result = subprocess.run(
        get_mpi_command(processes, script),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT
    )

    assert result.returncode == 0, (
        f"MPI-программа завершилась с ошибкой.\n"
        f"Код возврата: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    return result.stdout


@pytest.fixture
def temporary_project(tmp_path):
    """
    Создаёт временную структуру проекта.

    Не изменяет настоящие исходные файлы и данные.
    """

    src_dir = tmp_path / "src"
    data_dir = tmp_path / "data"

    src_dir.mkdir()
    data_dir.mkdir()

    return src_dir, data_dir


# ============================================================
# ЧАСТЬ 1. СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ
# ============================================================

@pytest.mark.parametrize("processes", [2, 4])
def test_parallel_dot_product(processes, temporary_project):
    """
    Проверка параллельного вычисления скалярного произведения.

    Проверяется:
    1. Успешный запуск MPI-программы.
    2. Правильность параллельного результата.
    3. Совпадение с последовательным результатом.
    4. Наличие сообщения об успешной проверке.
    """

    src_dir, _ = temporary_project

    script = src_dir / "parallel_dot_product.py"

    shutil.copy2(DOT_SCRIPT, script)

    output = run_mpi(
        processes=processes,
        script=script,
        cwd=src_dir
    )

    # Проверяем, что программа вывела оба результата.
    parallel_match = re.search(
        r"Параллельный результат:\s*([-+0-9.eE]+)",
        output
    )

    sequential_match = re.search(
        r"Последовательный результат:\s*([-+0-9.eE]+)",
        output
    )

    assert parallel_match is not None, (
        "Не найден параллельный результат."
    )

    assert sequential_match is not None, (
        "Не найден последовательный результат."
    )

    parallel_result = float(parallel_match.group(1))
    sequential_result = float(sequential_match.group(1))

    # Для вектора [1, 2, ..., 20]:
    # 1^2 + 2^2 + ... + 20^2 = 2870.
    expected = np.dot(
        np.arange(1, 21, dtype=np.float64),
        np.arange(1, 21, dtype=np.float64)
    )

    # Проверяем математическую правильность.
    assert np.isclose(
        parallel_result,
        expected,
        rtol=1e-10,
        atol=1e-10
    ), (
        f"Неправильное скалярное произведение: "
        f"{parallel_result}, ожидалось {expected}"
    )

    # Проверяем совпадение параллельного
    # и последовательного результатов.
    assert np.isclose(
        parallel_result,
        sequential_result,
        rtol=1e-10,
        atol=1e-10
    ), "Параллельный и последовательный результаты различаются."

    # Проверяем сообщение программы.
    assert "Проверка пройдена" in output


# ============================================================
# ЧАСТЬ 2. УМНОЖЕНИЕ ТРАНСПОНИРОВАННОЙ МАТРИЦЫ НА ВЕКТОР
# ============================================================

def prepare_matvec_data(data_dir, A, x):
    """
    Создаёт входные файлы для умножения A.T @ x.

    Формат in.dat:
        N
        M
    """

    M, N = A.shape

    np.savetxt(data_dir / "AData.dat", A)
    np.savetxt(data_dir / "xData.dat", x)

    with open(data_dir / "in.dat", "w") as file:
        file.write(f"{N}\n")
        file.write(f"{M}\n")


def _run_parallel_matvec_case(
    processes,
    M,
    N,
    temporary_project
):
    """
    Проверяет умножение транспонированной матрицы на вектор.

    Создаёт тестовые данные, запускает MPI-программу
    и сравнивает результат с независимым вычислением NumPy.
    """

    src_dir, data_dir = temporary_project

    script = src_dir / "parallel_matvec.py"

    shutil.copy2(MATVEC_SCRIPT, script)

    # Создаём воспроизводимые тестовые данные.
    rng = np.random.default_rng(seed=42 + M + N)

    A = rng.uniform(-10, 10, size=(M, N)).astype(np.float64)

    x = rng.uniform(-10, 10, size=M).astype(np.float64)

    # Сохраняем входные данные.
    prepare_matvec_data(data_dir, A, x)

    # Независимый последовательный результат.
    expected = np.dot(A.T, x)

    # Запускаем программу из папки src.
    # Благодаря этому ../data указывает на временную папку data.
    output = run_mpi(
        processes=processes,
        script=script,
        cwd=src_dir
    )

    # Извлекаем параллельный результат.
    parallel_match = re.search(
        r"Параллельный результат:\s*\[([^\]]+)\]",
        output
    )

    sequential_match = re.search(
        r"Последовательный результат:\s*\[([^\]]+)\]",
        output
    )

    assert parallel_match is not None, (
        f"Не найден параллельный результат.\nВывод:\n{output}"
    )

    assert sequential_match is not None, (
        f"Не найден последовательный результат.\nВывод:\n{output}"
    )

    parallel_result = np.fromstring(
        parallel_match.group(1),
        sep=" "
    )

    sequential_result = np.fromstring(
        sequential_match.group(1),
        sep=" "
    )

    # Проверяем размер результирующего вектора.
    assert parallel_result.shape == (N,), (
        f"Неправильный размер параллельного результата: "
        f"{parallel_result.shape}, ожидался {(N,)}"
    )

    assert sequential_result.shape == (N,), (
        f"Неправильный размер последовательного результата: "
        f"{sequential_result.shape}, ожидался {(N,)}"
    )

    # Проверяем, что параллельный результат совпадает
    # с независимым вычислением NumPy.
    assert np.allclose(
        parallel_result,
        expected,
        rtol=1e-10,
        atol=1e-10
    ), (
        f"Неправильный параллельный результат.\n"
        f"Получено: {parallel_result}\n"
        f"Ожидалось: {expected}"
    )

    # Проверяем последовательный результат программы.
    assert np.allclose(
        sequential_result,
        expected,
        rtol=1e-10,
        atol=1e-10
    ), "Последовательный результат программы неправильный."

    # Проверяем совпадение двух результатов программы.
    assert np.allclose(
        parallel_result,
        sequential_result,
        rtol=1e-10,
        atol=1e-10
    ), "Параллельный и последовательный результаты различаются."

    assert "Проверка пройдена" in output


# ============================================================
# ТЕСТИРОВАНИЕ РАЗЛИЧНЫХ РАЗМЕРОВ МАТРИЦЫ
# ============================================================

@pytest.mark.parametrize(
    "processes, M, N",
    [
        # Два процесса: один рабочий процесс.
        (2, 5, 3),
        (2, 7, 4),

        # Четыре процесса: три рабочих процесса.
        # M не делится на 3.
        (4, 5, 3),
        (4, 7, 4),

        # Проверка нулевого блока строк:
        # M меньше количества рабочих процессов.
        (4, 2, 3),
    ]
)
def test_parallel_matvec(
    processes,
    M,
    N,
    temporary_project
):
    """
    Проверка умножения A.T @ x на разных размерах данных
    и разном количестве MPI-процессов.
    """

    _run_parallel_matvec_case(
        processes=processes,
        M=M,
        N=N,
        temporary_project=temporary_project
    )
