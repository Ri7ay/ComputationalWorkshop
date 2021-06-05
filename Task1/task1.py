import numpy as np


def norm(A: np.array) -> float:
    res: float = 0
    for column in A:
        for elem in column:
            res += elem ** 2
    return res ** 0.5


def cond(A: np.array) -> float:
    A_reverse = np.linalg.inv(A)
    return norm(A_reverse) * norm(A)


def dx(x: np.array, x2: np.array) -> float:
    norma_X: float = (x[0] ** 2 + x[1] ** 2) ** 0.5
    normaXX2: float = ((x2[0] - x[0]) ** 2 + (x2[1] - x[1]) ** 2) ** 0.5
    return normaXX2 / norma_X


def estimate(A, b, b2) -> float:
    norma_delta_b: float = ((b2[0] - b[0]) ** 2 + (b2[1] - b[1]) ** 2) ** 0.5
    norma_b: float = (b[0] ** 2 + b[1] ** 2) ** 0.5
    return cond(A) * (norma_delta_b / norma_b)


def swap_matrix_row(A: np.array, p: int, q: int):
    A[:, [p, q]] = A[:, [q, p]]


def larger_matrix_element(A: np.array, p: int) -> int:
    i: int = 0
    if A[p][1] > A[p][i]:
        i = 1
    if A[p][2] > A[p][i]:
        i = 2
    return i


def Gauss_method(A: np.array, b: np.array) -> np.array:
    index_of_larger = larger_matrix_element(A, 0)
    if index_of_larger != 0:
        swap_matrix_row(A, index_of_larger, 0)
    tmp = A[1][0] / A[0][0]
    for i in range(3):
        A[1][i] -= A[0][i] * tmp
    b[1] -= b[0] * tmp
    tmp = A[2][0] / A[0][0]
    for i in range(3):
        A[2][i] -= A[0][i] * tmp
    b[2] -= b[0] * tmp
    tmp = A[2][1] / A[1][1]
    for i in range(3):
        A[2][i] -= A[1][i] * tmp
    b[2] -= b[1] * tmp
    b[2] /= A[2][2]
    x3 = b[2]
    b[1] = (b[1] - x3 * A[1][2]) / A[1][1]
    x2 = b[1]
    b[0] = (b[0] - x2 * A[0][1] - x3 * A[0][2]) / A[0][0]
    x1 = b[0]
    return np.array([x1, x2, x3])


def det_LU(A: np.array):
    L = np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]])
    U = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])

    L[0][0] = A[0][0]
    L[1][0] = A[1][0]
    L[2][0] = A[2][0]
    U[0][1] = A[0][1] / L[0][0]
    U[0][2] = A[0][2] / L[0][0]
    L[1][1] = A[1][1] - L[1][0] * U[0][1]
    U[1][2] = (A[1][2] - L[1][0] * U[0][2]) / L[1][1]
    L[2][1] = A[2][1] - L[2][0] * U[0][1]
    L[2][2] = A[2][2] - L[2][0] * U[0][2] - L[2][1] * U[1][2]
    return L[0][0] * L[1][1] * L[2][2]


def inverse_Jordan(A: np.array) -> np.array:
    result = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]])

    tmp: float = A[1][0] / A[0][0]
    for i in range(3):
        A[1][i] -= A[0][i] * tmp
        result[1][i] -= result[0][i] * tmp

    tmp: float = A[2][0] / A[0][0]
    for i in range(3):
        A[2][i] -= A[0][i] * tmp
        result[2][i] -= result[0][i] * tmp

    tmp: float = A[2][1] / A[1][1]
    for i in range(3):
        A[2][i] -= A[1][i] * tmp
        result[2][i] -= result[1][i] * tmp

    for i in range(3):
        tmp = A[i][i]
        for j in range(3):
            A[i][j] /= tmp
            result[i][j] /= tmp

    tmp: float = A[1][2] / A[2][2]
    for i in range(3):
        A[1][i] -= A[2][i] * tmp
        result[1][i] -= result[2][i] * tmp

    tmp: float = A[0][2] / A[2][2]
    for i in range(3):
        A[0][i] -= A[2][i] * tmp
        result[0][i] -= result[2][i] * tmp

    tmp: float = A[0][1] / A[1][1]
    for i in range(3):
        A[0][i] -= A[1][i] * tmp
        result[0][i] -= result[1][i] * tmp
    return result


def first_task():
    A = np.array([[-402.9, 200.7], [1204.2, -603.6]])
    b = np.array([200, -600])
    b2 = np.array([199, -601])
    x = np.linalg.solve(A, b)
    x2 = np.linalg.solve(A, b2)

    print("10.4\nВариант 7", end='\n--------------------------------------\n')
    print('A = ', A, sep='\n', end='\n\n')
    print('b = ', b, '\n', 'b\' = ', b2, '\n', 'x = ', x, '\n', 'x\' = ', x2, sep='', end='\n\n')
    print('cond(A) = ', cond(A))
    print('dx = ', dx(x, x2))

    if dx(x, x2) <= estimate(A, b, b2):
        print("dx <= ", estimate(A, b, b2))
    print(end='\n\n')


def second_task():
    A = np.array([[9.331343, 1.120045, -2.880925],
                  [1.120045, 7.086042, 0.670297],
                  [-2.880925, 0.670297, 5.622534]])
    b = np.array([7.570463, 8.876384, 3.411906])

    print("11.6\nВариант 7", end='\n--------------------------------------\n')
    print('b =', b)
    print('det(A) = ', det_LU(A.copy()))
    print('A^(-1) = ', inverse_Jordan(A.copy()), sep='\n')
    print('x = ', Gauss_method(A.copy(), b))


def main():
    first_task()
    second_task()


if __name__ == '__main__':
    main()
