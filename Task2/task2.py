import numpy as np


def swap_matrix_row(A: np.array, p: int, q: int):
    A[:, [p, q]] = A[:, [q, p]]


def larger_matrix_element(A: np.array, p: int) -> int:
    tmp = 0
    if A[p][1] > A[p][tmp]:
        tmp = 1
    if A[p][2] > A[p][tmp]:
        tmp = 2
    return tmp


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


def D_matrix(A: np.array) -> np.array:
    D = np.copy(A)
    for i in range(3):
        for j in range(3):
            if i != j:
                D[i][j] = 0
    return D


def H_D_matrix(A: np.array) -> np.array:
    D = D_matrix(A)
    E = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    return E - np.dot(np.linalg.inv(D), A)


def g_D_matrix(A: np.array, b: float) -> np.array:
    D = D_matrix(A)
    return np.dot(np.linalg.inv(D), b)


def aprior_estimate(H: float, g: float) -> float:
    norm_H = np.linalg.norm(H, np.inf)
    norm_g = np.linalg.norm(g, np.inf)
    return (norm_H ** 7) / (1. - norm_H) * norm_g


def aposterior_estimate(H: float, g: float, xk: float, xk_1: float) -> float:
    norm_H = np.linalg.norm(H, np.inf)
    norm_g = np.linalg.norm(g, np.inf)
    norm_x = np.linalg.norm(xk - xk_1, np.inf)
    return norm_H / (1. - norm_H) * norm_x


def lusteric_refinement(H: float, xk: float, xk_1: float) -> float:
    sp_radius = spectr_radius(H.copy())
    return xk_1 + 1. / (1. - sp_radius) * (xk - xk_1)


def simple_iteration_method(H: float, g: float, k: int) -> np.array:
    x_i = np.array([0., 0., 0.])
    for j in range(1, k + 1):
        x_i = np.dot(H, x_i) + g
    return x_i


def spectr_radius(A: np.array) -> float:
    eigenvalue, eigenvector = np.linalg.eigh(A)
    res = abs(eigenvalue[0])
    for i in eigenvalue:
        if res < abs(i):
            res = abs(i)
    return res


def pre_func(H: np.array) -> np.array:
    Hl = H.copy()
    Hr = H.copy()
    for i in range(3):
        for j in range(3):
            if i == j:
                Hl[i][j] = 0
            if i < j:
                Hl[i][j] = 0
            if i > j:
                Hr[i][j] = 0
    return [Hl, Hr]


def seidel_method(H: np.array, g: np.array, k: int) -> np.array:
    Hl, Hr = pre_func(H)
    x_i = np.array([0., 0., 0.])
    E = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])
    for i in range(1, k + 1):
        x_i = np.dot(np.dot(np.linalg.inv(E - Hl), Hr), x_i) + np.dot(np.linalg.inv(E - Hl), g)
    sp_radius = spectr_radius(np.dot(np.linalg.inv(E - Hl), Hr))
    return x_i, sp_radius


def relaxation_method(H: np.array, g: float, k: int):
    Hl, Hr = pre_func(H)
    x_i = np.array([0., 0., 0.])
    q = 2. / (1 + (1 - spectr_radius(H) ** 2) ** 0.5)
    E = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])
    temp0 = np.linalg.inv(E - q * Hl)
    temp1 = np.dot((1 - q) * E + q * Hr, temp0)
    temp2 = np.dot(g, q * temp0)
    for i in range(1, k + 1):
        x_i = np.dot(x_i, temp1) + temp2
    return x_i


def main():
    print('Лабораторная работа 2\nвариант 7')
    A = np.array([[9.331343, 1.120045, -2.880925],
                  [1.120045, 7.086042, 0.670297],
                  [-2.880925, 0.670297, 5.622534]])

    b = np.array([7.570463, 8.876384, 3.411906])
    x = Gauss_method(A.copy(), b.copy())
    print('1)Решение методом Гаусса: x* = ', x)
    H_D = H_D_matrix(A.copy())
    g_D = g_D_matrix(A.copy(), b.copy())
    print('Преобразовали систему Ax=b к виду x = HD * x + gD, где\nHD =\n', H_D, '\ngD = ', g_D)

    norm_H_D = np.linalg.norm(H_D.copy(), np.inf)
    print('||HD|| = ', norm_H_D)

    print('Aприорная оценка погрешности ||x^7-x*|| = ', aprior_estimate(H_D.copy(), g_D.copy()))

    x7 = simple_iteration_method(H_D.copy(), g_D.copy(), 7)
    x6 = simple_iteration_method(H_D.copy(), g_D.copy(), 6)
    x_lusteric = lusteric_refinement(H_D.copy(), x7, x6)
    print('4)Метод простой итерации, x^7 = ', x7)
    print('Фактическая погрешность ||x^7 - x*|| = ', np.linalg.norm((x7 - x).copy(), np.inf))
    print('Априорная оценка погрешности:     ', aprior_estimate(H_D.copy(), g_D.copy()))
    print('Апостериорная оценка погрешности: ', aposterior_estimate(H_D.copy(), g_D.copy(), x7, x6))
    print('Уточнение последнего приближения по Люстернику: ', x_lusteric)
    print('Фактическая погрешность приближения по Люстерику: ', np.linalg.norm((x_lusteric - x).copy(), np.inf))

    x7_seidel, sp_radius = seidel_method(H_D.copy(), g_D.copy(), 7)
    print('Метод Зейдела, x^7 = ', x7_seidel)
    print('Фактическая погрешность: ', np.linalg.norm((x7_seidel - x).copy(), np.inf))
    if np.linalg.norm((x7_seidel - x).copy(), np.inf) < np.linalg.norm((x7 - x).copy(), np.inf):
        print('Метод Зейделя точнее, чем метод простой итерации')
    else:
        print('Метод простой итерации точнее, чем метод Зейделя')

    print('6)Спектральный радиус матрицы перехода в методе Зейделя', sp_radius)

    x7_relaxation = relaxation_method(H_D.copy(), g_D.copy(), 7)
    print('7) Метод релаксации, x^7 = ', x7_relaxation)

    print('Сравнение фактических погрешностей разных методов')
    best_method = 'метод простой итерации'
    val_min = np.linalg.norm((x7 - x).copy(), np.inf)

    if np.linalg.norm((x_lusteric - x).copy(), np.inf) < val_min:
        val_min = np.linalg.norm((x_lusteric - x).copy(), np.inf)
        best_method = 'метод приближения по Люстерику'
    if np.linalg.norm((x7_seidel - x).copy(), np.inf) < val_min:
        val_min = np.linalg.norm((x7_seidel - x).copy(), np.inf)
        best_method = 'метод Зейделя'
    if np.linalg.norm((x7_relaxation - x).copy(), np.inf) < val_min:
        val_min = np.linalg.norm((x7_relaxation - x).copy(), np.inf)
        best_method = 'метод релаксации'
    print('Метод простой итерации:   ', np.linalg.norm((x7 - x).copy(), np.inf))
    print('Приближение по Люстерику: ', np.linalg.norm((x_lusteric - x).copy(), np.inf))
    print('Метод Зейделя:            ', np.linalg.norm((x7_seidel - x).copy(), np.inf))
    print('Метод релаксации:         ', np.linalg.norm((x7_relaxation - x).copy(), np.inf))
    print('Самым точным оказался ', best_method)


if __name__ == '__main__':
    main()
