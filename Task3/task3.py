import numpy as np


def eigen_sort(eigen_value, eigen_vector):
    n = len(eigen_value)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(eigen_value[i]) < abs(eigen_value[j]):
                eigen_value[i], eigen_value[j] = eigen_value[j], eigen_value[i]
                for k in range(n):
                    eigen_vector[i][k], eigen_vector[j][k] = eigen_vector[j][k], eigen_vector[i][k]


def power_method(a_matrix, eps):
    print('Степенной метод')
    eigen_value, eigen_vector = np.linalg.eig(a_matrix)
    eigen_sort(eigen_value, eigen_vector)

    y_matrix = np.asarray(eigen_vector[0])
    prev_eigen = 1.0
    k = 0
    curr_eigen = 0

    while abs(prev_eigen - eigen_value[0]) > eps:
        tmp_matrix = np.dot(a_matrix, y_matrix)
        for i in range(len(y_matrix)):
            if abs(prev_eigen - eigen_value[0]) > abs(tmp_matrix[i] / y_matrix[i] - eigen_value[0]):
                curr_eigen = tmp_matrix[i] / y_matrix[i]
        k += 1
        print(f'k = {k} lambda_k = {round(curr_eigen, 12)} \nlambda_k - lambda_k-1 = {curr_eigen - prev_eigen}')
        norm_tmp = np.linalg.norm(np.dot(a_matrix, y_matrix / 10 ** k) - curr_eigen * (y_matrix / 10 ** k))
        print(f'|lambda_k - lambda*| = {abs(curr_eigen - eigen_value[0])} \n|A * x_k - lambda_k * x_k| = {norm_tmp}')
        norm_tmp = np.linalg.norm(a_matrix.dot(y_matrix / 10 ** k) - curr_eigen *
                                   (y_matrix / 10 ** k)) / np.linalg.norm(y_matrix / 10 ** k)
        print(f'Апостериорная оценка {norm_tmp}')
        y_matrix = tmp_matrix
        prev_eigen = curr_eigen
        print()


def scalar_product_method(a_matrix, epsilon):
    print('Метод скалярных произведений\n')

    eigen_value, eigen_vector = np.linalg.eig(a_matrix)
    eigen_sort(eigen_value, eigen_vector)

    y_matrix = np.asarray(eigen_vector[0])
    eigen_prev = 1.0
    k = 0
    while abs(eigen_prev - eigen_value[0]) > epsilon ** 2:
        tmp_matrix = np.dot(a_matrix, y_matrix)
        eigen_tmp = np.dot(tmp_matrix, y_matrix) / np.dot(y_matrix, y_matrix)
        k += 1
        print(f'k = {k} \nlambda_k = {round(eigen_tmp, 12)} + \nlambda_k - lambda_k-1 = {eigen_tmp - eigen_prev}')
        norm_tmp = np.linalg.norm(a_matrix.dot(y_matrix / 10 ** k) - eigen_tmp * (y_matrix / 10 ** k))
        print(f'|lambda_k - lambda*| = {abs(eigen_tmp - eigen_value[0])} \n|A * x_k - lambda_k * x_k| = {norm_tmp}')
        norm_tmp = np.linalg.norm(np.dot(a_matrix, y_matrix / 10 ** k) - eigen_tmp * (y_matrix / 10 ** k)) / \
                   np.linalg.norm(y_matrix / 10 ** k)
        print('Апостериорная оценка ' + str(norm_tmp) + '\n')
        y_matrix = tmp_matrix
        eigen_prev = eigen_tmp


def main():
    a_matrix = np.array([[9.331343, 1.120045, -2.880925],
                  [1.120045, 7.086042, 0.670297],
                  [-2.880925, 0.670297, 5.622534]])
    epsilon = 0.001
    print('Задача нахождения максимального по модулю собсвтенного числа матрицы')
    print('Вариант 7')
    print('\nИсходная матрица A: ', a_matrix, sep='\n')

    eigen_value, eigen_vector = np.linalg.eig(a_matrix)
    eigen_sort(eigen_value, eigen_vector)

    print(f'\nНаибольшое по модулю собственное число = {eigen_value[0]}')
    print(f'Отношение lambda1 / lambda2 = {eigen_value[1] / eigen_value[0]}')

    power_method(a_matrix, epsilon)
    scalar_product_method(a_matrix, epsilon)


if __name__ == '__main__':
    main()