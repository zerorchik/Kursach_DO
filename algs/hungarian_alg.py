# -*- coding: utf-8 -*-

import numpy as np
import copy

def solve_hungarian(f, v, p):
    f_copy = copy.deepcopy(f)  # create a copy of the matrix
    selected = [[] for _ in range(n)]  # list of selected candidates for each vacancy
    max_f = [[] for _ in range(n)]
    max_f_counts = [0] * len(f_copy[0])  # initialize counter for each column
    stop = False

    # Вхідна задача
    for i in range(len(v)):
        print(v[i], '\t', p[i], '\t', f_copy[i])
    selected = [[] for _ in range(n)]
    print()

    for i in range(n):
        for j in range(m):
            if f_copy[i][j] == 0:
                f_copy[i][j] = -np.inf  # exclude zeros
    print(f'без нулів:\n{f_copy}')

    counter = 0
    while not stop:
        counter += 1
        print(f'\nКрок {counter}')
        # Якщо на мінімум
        if counter > 1:
            for i in range(n):
                for j in range(m):
                    if f_copy[i][j] != -np.inf and f_copy[i][j] != 0:
                        f_copy[i][j] = 1 - f_copy[i][j]
            print(f'перевод на максимум')
        print(f'f_copy:\n{f_copy}')

        all_is_null = False
        for i in range(n):
            max_val = round(np.max(f_copy[i]), 2)
            max_index = np.argmax(f_copy[i])
            if max_val != -np.inf:
                max_f[i].append(max_val)  # maximum value for each vacancy
                print(f'max_f[{i}] = {max_f[i][0]}')
                max_f_counts[max_index] += 1
        print(f'кількість вибраних елементів на кандидата = {max_f_counts}')
        if all(value == 0 for sublist in max_f for value in sublist):
            all_is_null = True

        if all_is_null:
            max_f = [[] for _ in range(n)]
            for j in range(m):
                if max_f_counts[j] == n:
                    max_f_counts = [0] * len(f_copy[0])
                    print('Вибрані елементи знаходяться в одному рядку')
                    max_val = round(np.max(f[j]), 2)
                    max_f[i].append(max_val)  # maximum value for each vacancy
                    print(f'max_f[{i}] = {max_f[i][0]}')
                    max_f_counts[max_index] += 1

        for i in range(n):
            if len(max_f[i]) > 0:
                for j in range(m):
                    if f_copy[i][j] != -np.inf and f_copy[i][j] != 0:
                        f_copy[i][j] = round(max_f[i][0] - f_copy[i][j], 2)  # subtraction
                print(f'{max_f[i][0]} - f_copy[{i}] = {f_copy[i]}')

        for i in range(n):
            for j in range(m):
                if f_copy[i][j] == 0 and max_f_counts[j] == 1:
                    print(f'{f[i][j]} - єдиний макс елемент у стовпці')
                    if v[i] > 0:  # check quota
                        print('квота виконується')
                        selected[i].append(j)  # select candidate
                        print(f'selected:\n{selected}')
                        v[i] -= 1  # update quota
                        print(f'нова квота\n{v}')
                        print('кандидата назначено')
                        f_copy[:, j] = -np.inf  # mark as considered
                        print(f_copy)
            if v[i] == 0:
                print('вакансію закрито')
                f_copy[i, :] = -np.inf  # mark as considered
                print(f_copy)

        max_f = [[] for _ in range(n)]
        max_f_counts = [0] * len(f_copy[0])

        if np.all(v == 0) or np.all(f_copy == -np.inf):
            stop = True

        if counter == 100 * max(n, m):
            print('------------------------------------')
            print('Задача не зійшлась')
            print('------------------------------------')
            stop = True


    selected_values = [[] for _ in range(n)]
    for i in range(n):
        for j in selected[i]:
            selected_values[i].append(f[i][j])

    z = 0
    real_z = 0
    for i in range(n):
        z += round(np.sum(selected_values[i]) * p[i], 2)
        real_z += round(np.sum(selected_values[i]), 2)
    z = round(z, 2)

    ideal_z = 0
    for j in range(m):
        x = max(row[j] for row in f)
        ideal_z += x
        ideal_z = round(ideal_z, 2)
    y = round(real_z / ideal_z, 2)

    print(f'\nРозподіл:\n{selected_values}')
    print(f'z = {z}')
    print(f'y = {y}')

    return selected_values, z, y

def read_individual_task(filename):
    with open(filename, 'rb') as f_in:
        m = np.load(f_in)
        n = np.load(f_in)
        f_matrix = np.load(f_in, allow_pickle=True)
        v_list = np.load(f_in, allow_pickle=True).tolist()
        p_list = np.load(f_in, allow_pickle=True).tolist()
    return m, n, f_matrix, v_list, p_list


m, n, f_matrix, v_list, p_list = read_individual_task('task_m_3_n_3_f_mean_0.5_f_var_0.25_v_mean_3_v_var_0.5_p_mean_0.5_p_var_0.25.txt')
# # Захардкоджені параметри
# f_matrix = np.array([[0.96, 0.91, 0.8, 0, 0.97],
#               [0, 0.7, 0.51, 0.72, 0],
#               [0.5, 0, 0.6, 0.6, 0.99]])
# v_list = [2, 1, 2]  # quotas for each vacancy
# p_list = [0.5, 0.3, 0.2]  # priorities for each vacancy
# n = 3
# m = 5

f_rez, z, y = solve_hungarian(f_matrix, v_list, p_list)
