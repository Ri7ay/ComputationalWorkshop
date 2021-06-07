import numpy as np
import pandas as pd


def a(x_value: float) -> float:
    return np.cos(x_value)


def phi(x_value: np.array) -> np.array:
    return x_value ** 3


def alpha(t_value: float) -> float:
    return t_value ** 3


def beta() -> int:
    return 3


def f(x_value: float, t_value: float) -> float:
    return x_value * (t_value ** 2)


def L(i: int, k: int, u: list):
    return a(x[i]) * ((u[k][i + 1] - 2 * u[k][i] + u[k][i - 1]) / (h ** 2))


# Явная разностная схема
def explicit_difference_scheme() -> np.array:
    solutions = []

    u_0i = phi(x)
    solutions.append(u_0i)

    for k in range(1, M + 1):
        u_k = [solutions[k - 1][i] + tau * (L(i, k - 1, solutions) + f(x[i], t[k - 1])) for i in range(1, N)]

        u_0k = alpha(t[k])
        u_k.insert(0, u_0k)

        u_nk = 2 / 3 * h * beta() + 4 * u_k[N - 1] + 2 * u_k[N - 2]
        u_k.insert(N, u_nk)

        u_copy = np.copy(u_k)
        solutions.append(u_copy)
        u_k.clear()

    return solutions


# Схема с весами
def weights_scheme() -> np.array:
    solutions = []

    u_0i = phi(x)
    solutions.append(u_0i)

    for k in range(1, M + 1):

        # Реализуем трехдиагональную матрицу m
        m = np.zeros((N + 1, N + 1))

        # Первыя ряд
        m[0][0] = 1
        m[0][1] = 0

        # Последний ряд
        m[N][N - 1] = -1 / h
        m[N][N] = -1 / h

        # Столбец справа
        g = np.zeros(N + 1)
        g[0] = alpha(t[k])
        g[N] = beta()

        for i in range(1, N):
            m[i][i - 1] = a(x[i]) / h ** 2
            m[i][i] = -2 * a(x[i]) / h ** 2
            m[i][i + 1] = a(x[i]) / h ** 2
            g[i] = -1 / tau * solutions[k - 1][i] - (1 - 1 / 2) * L(i, k - 1, solutions) - f(x[i], t[k])

        solutions.append(np.linalg.solve(m, g))

    return solutions


if __name__ == '__main__':
    N = 5
    h = 1 / N
    x = np.arange(0, 1 + h, h)

    T = 1
    M = 5
    tau = T / M
    t = np.arange(0, T + tau, tau)

    first = pd.DataFrame(weights_scheme(), x, columns=["u_0", "u_1", "u_2", "u_3", "u_4", "u_5"]).T.round(5)
    second = pd.DataFrame(explicit_difference_scheme(), x,
                              columns=["u_0", "u_1", "u_2", "u_3", "u_4", "u_5"]).T.round(5)

    print(first, second, sep='\n')
