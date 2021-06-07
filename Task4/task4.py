from math import cosh
import numpy as np
import pandas as pd


def H(x: float, y: float) -> float:
    return cosh(x * y)


def f(x: np.array) -> np.array:
    return x - 0.6


def middle_rectangles(N: int, a: float, b: float) -> float:
    h = (b - a) / N

    x = np.array([a + (h / 2) + (k - 1) * h for k in range(1, N + 1)])
    y = f(x)
    result = h * sum(y)

    return result


def middle_rectangles_points_coefficents(N: int, a: float, b: float) -> [np.array, np.array]:
    h = (b - a) / N

    points = np.array([a + (h / 2) + (k - 1) * h for k in range(1, N + 1)])
    coefficents = [h] * N

    return points, coefficents


def kroneker_delta(x: float, y: float) -> bool:
    return True if x == y else False


def mechanical_quadrature(N: int, find_points_coefficents: np.array, x: float, c: float, a: float, b: float) -> float:
    points, coeff = find_points_coefficents(N, a, b)

    # Решение системы Dz=g
    D = np.zeros((N, N))
    for row in range(N):
        for col in range(N):
            D[row][col] = kroneker_delta(row, col) - coeff[col] * H(points[row], points[col])
    g = f(points)
    z = np.linalg.solve(D, g)

    sum = 0
    for i in range(N):
        sum += coeff[i] * H(x, points[i]) * z[i]

    result = sum + f(x)
    return result


if __name__ == '__main__':
    c = 0.6
    a, b = [0, 1]

    # Начальное число разбиений
    n = 5
    start_calc = 7
    splits = [2 ** i * n for i in range(start_calc)]

    # Первая колонка таблицы
    x_s = list(map(lambda x: 'u^({})(x)'.format(x), splits))
    # Вычисление значение в точке a
    a_s = [mechanical_quadrature(n, middle_rectangles_points_coefficents, a, c, a, b) for n in splits]
    # Вычисление значение в точке (a+b)/2
    a_b_s = [mechanical_quadrature(n, middle_rectangles_points_coefficents, (a + b) / 2, c, a, b) for n in splits]
    # Вычисление значение в точке b
    b_s = [mechanical_quadrature(n, middle_rectangles_points_coefficents, b, c, a, b) for n in splits]
    # Заполнение таблицы
    data = pd.DataFrame(list(zip(x_s, a_s, a_b_s, b_s)), columns=['x', 'a', '(a + b)/2', 'b'])

    data.set_index('x', inplace=True)
    print(data)
