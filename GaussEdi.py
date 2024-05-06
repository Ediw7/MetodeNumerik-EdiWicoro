import numpy as np
import unittest

# Fungsi Dekomposisi LU menggunakan metode eliminasi Gauss
def lu_decomposition_gauss(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Mengisi bagian diagonal L dengan 1
        L[i][i] = 1

        # Menghitung elemen-elemen U
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = matrix[i][k] - sum

        # Menghitung elemen-elemen L
        for k in range(i + 1, n):
            sum = 0
            for j in range(i):
                sum += (L[k][j] * U[j][i])
            L[k][i] = (matrix[k][i] - sum) / U[i][i]

    return L, U

# Menyelesaikan sistem persamaan linear dengan Dekomposisi LU
def solve_lu_decomposition(A, b):
    L, U = lu_decomposition_gauss(A)
    n = len(A)
    # Substitusi maju untuk mencari y
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    # Substitusi mundur untuk mencari x
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

# Soal yang diberikan
A = np.array([[1,1, 1], [1, 2, -1], [2, 1, 2]])
b = np.array([6, 2, 10])

class TestLU(unittest.TestCase):
    def test_solve_lu_decomposition(self):
        # Expected solution
        expected_solution = np.array([1, 2, 3])
        # Compute the solution using LU decomposition
        computed_solution = solve_lu_decomposition(A, b)
        # Check if the computed solution matches the expected solution
        np.testing.assert_array_almost_equal(computed_solution, expected_solution)

if __name__ == '__main__':
    unittest.main()
