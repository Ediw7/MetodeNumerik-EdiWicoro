import numpy as np
import unittest

# Fungsi untuk mencari matriks balikan menggunakan NumPy


def inverse_matrix(matrix):
    try:
        inverse = np.linalg.inv(matrix)
        return inverse
    except np.linalg.LinAlgError:
        return None


# Contoh penggunaan
A = np.array([[1, -1, 2], [3, 0, 1], [1, 0, 2]])
inverse_A = inverse_matrix(A)
if inverse_A is not None:
    print("Matriks Balikan (inverse) A:")
    print(inverse_A)
else:
    print("Matriks A tidak memiliki balikan (inverse).")


# unit test
class TestInverseMatrix(unittest.TestCase):

    def test_inverse(self):
        # Tes untuk matriks yang memiliki balikan
        matrix = np.array([[1, -1, 2], [3, 0, 1], [1, 0, 2]])
        expected_result = np.array(
            [[0.0, 0.4, -0.2], [-1.0, 0.0, 1.0], [0.0, -0.2, 0.6]])
        self.assertTrue(np.allclose(inverse_matrix(matrix), expected_result))

    def test_singular_matrix(self):
        # Tes untuk matriks yang tidak memiliki balikan
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertIsNone(inverse_matrix(matrix))


if __name__ == '__main__':
    unittest.main()
