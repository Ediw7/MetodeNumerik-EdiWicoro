# Nama  : Edi Wicoro
# NIM  : 21120122130073
# Kelas  : Metode Numerik B / Teknik Komputer 


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
print("Matriks A:")
print(A)

# Langkah-langkah untuk mencari matriks balikan A
print("\nLangkah-langkah:")
det_A = np.linalg.det(A)
print("   Determinan matriks A =", det_A)
if det_A == 0:
    print("   Karena determinan A = 0, maka A tidak memiliki balikan (singular).")
else:
    adj_A = np.linalg.inv(A) * det_A
    print("   Matriks Adjoin A:")
    print(adj_A)
    inverse_A = inverse_matrix(A)
    print("   Matriks Balikan (inverse) A:")
    print(inverse_A)

# unit test
class TestInverseMatrix(unittest.TestCase):
    def test_inverse(self):
        # Tes untuk matriks yang memiliki balikan
        matrix = np.array([[1, -1, 2], [3, 0, 1], [1, 0, 2]])
        expected_result = np.array([[0.0, 0.4, -0.2], [-1.0, 0.0, 1.0], [0.0, -0.2, 0.6]])
        print("Expected Result:")
        print(expected_result)
        print("Actual Result:")
        print(inverse_matrix(matrix))
        self.assertTrue(np.allclose(inverse_matrix(matrix), expected_result))

    def test_singular_matrix(self):
        # Tes untuk matriks yang tidak memiliki balikan
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print("\nTest untuk matriks yang tidak memiliki balikan:")
        print("Expected Result: None")
        print("Actual Result:")
        print(inverse_matrix(matrix))
        self.assertIsNone(inverse_matrix(matrix))

if __name__ == '__main__':
    unittest.main()
