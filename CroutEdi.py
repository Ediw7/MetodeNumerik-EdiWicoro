# Nama  : Edi Wicoro
# NIM  : 21120122130073
# Kelas  : Metode Numerik B / Teknik Komputer 


import numpy as np
import unittest


def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j, j] = 1

        for i in range(j, n):
            sum_val = sum(L[i, k] * U[k, j] for k in range(i))
            L[i, j] = A[i, j] - sum_val

        for i in range(j, n):
            sum_val = sum(L[j, k] * U[k, i] for k in range(j))
            if L[j, j] == 0:
                return None, None  # Matriks tidak bisa didekomposisi
            U[j, i] = (A[j, i] - sum_val) / L[j, j]

    return L, U


# Contoh penggunaan
A = np.array([[2, 4, 3],
              [3, 5, 2],
              [4, 6, 3]])


L, U = crout_decomposition(A)
print("Matrix L:")
print(L)
print("Matrix U:")
print(U)


# A = np.array([[1, 1, -1], [-1, 1, 1], [2, 2, 1]])

# seharusnya U[[1, 1, -1], [0, 2, 0], [0, 0, 3]]
# seharusnya L[[1, 0, 0], [-1, 1, 0], [2, 0, 1]]

class TestCroutDecomposition(unittest.TestCase):
    def test_decomposition(self):
        A = np.array([[2, 4, 3],
                      [3, 5, 2],
                      [4, 6, 3]])
        expected_L = np.array([[2, 0, 0],
                               [3, -1, 0],
                               [4, -2, 2]])
        expected_U = np.array([[1, 2, 1.5],
                               [0, 1, 2.5],
                               [0, 0, 1]])
        L, U = crout_decomposition(A)
        np.testing.assert_array_almost_equal(L, expected_L)
        np.testing.assert_array_almost_equal(U, expected_U)


if __name__ == '__main__':
    unittest.main()
