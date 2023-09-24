import numpy as np
import math
import linalg
from typing import List, Callable
import time

def py_cosine_similarity(mat1: List[List[int]], mat2: List[List[int]]) -> List[int]:
    if len(mat1) != len(mat2):
        raise ValueError("Uncompatible set of vectors shapes")
    result = []
    for i in range(len(mat1)):
        norm1 = math.sqrt(sum([x ** 2 for x in mat1[i]]))
        norm2 = math.sqrt(sum([x ** 2 for x in mat2[i]]))
        dot_vectors = sum([mat1[i][j] * mat2[i][j] for j in range(len(mat1[0]))])
        if norm1 == 0 or norm2 == 0:
            result.append(1.0)
        else:
            result.append(dot_vectors / norm1 / norm2)
            
    return result

def numpy_cosine_similiraty(mat1: np.ndarray, mat2: np.ndarray) -> List[int]:
    if len(mat1) != len(mat2):
        raise ValueError("Uncompatible set of vectors shapes")
    cos_sim = np.dot(mat1, mat2.T).diagonal()/(np.linalg.norm(mat1, axis=1) * np.linalg.norm(mat2, axis=1))

    return cos_sim.tolist()

def test_timings(func: Callable, *args):
    _ = func(*args)
    start_time = time.time()
    _ = func(*args)
    end_time = time.time()
    return round(end_time - start_time, 5)


def compare(matrix_rows: int, matrix_columns: int) -> None:
    matrix_a = np.random.rand(matrix_rows, matrix_columns)
    matrix_b = np.random.rand(matrix_rows, matrix_columns)

    list_a = matrix_a.tolist()
    list_b = matrix_b.tolist()

    print(
        "Cos sim (Pure Python), size={0}x{1}: {2} seconds".format(
            matrix_rows, matrix_columns, test_timings(py_cosine_similarity, list_a, list_b)
        )
    )
    print(
        "Cos sim (C++ BLAS), size={0}x{1}: {2} seconds".format(
            matrix_rows, matrix_columns, test_timings(linalg.LinearAlgebra.cossimBlas, list_a, list_b)
        )
    )
    print(
        "Cos sim (Python numpy), size={0}x{1}: {2} seconds\n".format(
            matrix_rows, matrix_columns, test_timings(numpy_cosine_similiraty, np.array(list_a), np.array(list_b))
        )
    )

if __name__ == "__main__":
    rows = [10, 50, 100, 300, 500, 700]
    columns = [8, 10, 120, 200, 400, 1000]
    for size in zip(rows, columns):
        compare(*size)


