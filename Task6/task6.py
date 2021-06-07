import numpy as np
from prettytable import PrettyTable


# Возьмем системы ортогональных многочленов Якобы в качестве координатной системы
def JacobiPolynomial(k: float, n: float, x: float) -> np.array:
    if n == 0:
        return 1
    elif n == 1:
        return (1 + k) * x
    else:
        return ((n + k) * (2 * (n + k) - 1) * x * JacobiPolynomial(k, n - 1, x) - (n + k) * (n + k - 1) *
                JacobiPolynomial(k, n - 2, x)) / (n * (n + 2 * k))


def d_JacobiPolynomial(k: float, n: float, x: float) -> np.array:
    return 0 if n == 0 else (n + 2 * k + 1) * JacobiPolynomial(k + 1, n - 1, x) / 2


def omega(index: float, x: float) -> np.array:
    return (1 - x * x) * JacobiPolynomial(1, index, x)


def d_omega(index: float, x: float) -> np.array:
    return - 2 * (index + 1) * JacobiPolynomial(0, index + 1, x)


def dd_omega(index: float, x: float) -> np.array:
    return - 2 * (index + 1) * d_JacobiPolynomial(0, index + 1, x)


def legendPolynomial(n: float, x: float) -> np.array:
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendPolynomial(n - 1, x) - (n - 1) * legendPolynomial(n - 2, x)) / n


def d_legendPolynomial(n: float, x: float) -> np.array:
    if n == 0:
        return 0
    else:
        return n * (legendPolynomial(n - 1, x) - x * legendPolynomial(n, x)) / (1 - x * x)


def buildRoots(n: int, epsilon: float) -> np.array:
    result = []

    for i in range(1, n + 1):
        x = np.cos(np.pi * (4 * i - 1) / (4 * n + 2))
        prev = 2.
        while abs(x - prev) >= epsilon:
            prev = x
            x = prev - legendPolynomial(n, prev) / d_legendPolynomial(n, prev)
        result += [x]
    return result


def buildCoeff(n: int, roots: np.array) -> np.array:
    result = []
    for i in range(n):
        root = roots[i]
        result += [2 / ((1 - root * root) * ((d_legendPolynomial(n, root)) ** 2))]
    return result


def get_l_omegaScalarProduct(index: float) -> np.array:
    return lambda x: (x - 7) * dd_omega(index, x) / (8 + 3 * x) + (1 + x / 3) * d_omega(index, x) + \
                     (1 - np.exp(x / 2) / 2) * omega(index, x)


def multiply(f, g, rootsNumber: np.array, roots: np.array, coeff: np.array):
    result = 0

    for i in range(rootsNumber):
        root = roots[i]
        result += coeff[i] * f(root) * g(root)
    return result


def build_solution_least_square_method(value: int, f, rootsNumber: np.array, roots: np.array, coeff: np.array):
    l_omegas = []

    for i in range(value):
        l_omegas += [get_l_omegaScalarProduct(i)]

    g = np.zeros(value)
    for i in range(value):
        g[i] = multiply(f, l_omegas[i], rootsNumber, roots, coeff)

    A = np.zeros(value ** 2).reshape(value, value)
    for i in range(value):
        for j in range(value):
            A[i][j] = multiply(l_omegas[i], l_omegas[j], rootsNumber, roots, coeff)
    c = np.linalg.solve(A, g)
    return lambda x: sum([(c[i] * omega(i, x)) for i in range(value)])


def builtSolutionCollocationMethod(value: int, f):
    ch_roots = []
    for i in range(1, value + 1):
        ch_roots += [np.cos((2 * i - 1) * np.pi / 2 / value)]

    L_omegas = []
    for i in range(value):
        L_omegas += [get_l_omegaScalarProduct(i)]

    h = np.zeros(value)
    for i in range(value):
        h[i] = f(ch_roots[i])

    B = np.zeros(value ** 2).reshape(value, value)
    for i in range(value):
        for j in range(value):
            B[j][i] = L_omegas[i](ch_roots[j])

    c = np.linalg.solve(B, h)
    return lambda x: sum([(c[i] * omega(i, x)) for i in range(value)])


def main():
    rootsNumber = 10
    eps = 0.0001
    value = 7

    function = lambda x: 0.5 - x / 3

    roots = buildRoots(rootsNumber, eps)
    coefficients = buildCoeff(rootsNumber, roots)

    leastSquare = PrettyTable(
        ['n', 'y^n(-0.5)', 'y^n(0)', 'y^n(0.5)', '|y* - y^n(-0.5)|', '|y* - y^n(0)|', '|y* - y^n(0.5)|'])
    for i in range(1, value + 1):
        y = build_solution_least_square_method(i, function, rootsNumber, roots, coefficients)
        leastSquare.add_row(
            [i, y(-0.5), y(0), y(0.5), abs(0.156053 - y(-0.5)), abs(0.228862 - y(0)), abs(0.198417 - y(0.5))])
    print("Метод наименьших квадратов", leastSquare, sep='\n')

    table_collocation = PrettyTable(
        ['n', 'y^n(-0.5)', 'y^n(0)', 'y^n(0.5)', '|y* - y^n(-0.5)|', '|y* - y^n(0)|', '|y* - y^n(0.5)|'])
    for i in range(1, value + 1):
        y = builtSolutionCollocationMethod(i, function)
        table_collocation.add_row(
            [i, y(-0.5), y(0), y(0.5), abs(0.156053 - y(-0.5)), abs(0.228862 - y(0)), abs(0.198417 - y(0.5))])
    print("Метод колокации", table_collocation, sep='\n')


if __name__ == "__main__":
    main()
