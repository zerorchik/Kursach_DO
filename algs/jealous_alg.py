# -*- coding: utf-8 -*-

import numpy as np
import copy

def solve_greedy(m, n, f, v, p):
    f_copy = copy.deepcopy(f)
    v_copy = copy.deepcopy(v)
    p_copy = copy.deepcopy(p)

    # Вхідна задача
    for i in range(len(v)):
        print(v[i], '\t', p[i], '\t', f_copy[i])
    selected = [[] for _ in range(n)]
    print()

    for step in range(n * m):
        # Найбільший пріоритет
        i = np.argmax(p_copy)
        if len(selected[i]) < v[i]:
            x = np.argmax(f_copy[i])
            if f_copy[i][x] != 0:
                print(f'Крок {step + 1}:')
                print(f'пріоритет p = {p_copy[i]}')
                print(f'вакансія:\n{f_copy[i]}')
                print(f'найбільш підходящий кандидат {f_copy[i][x]} з індексом {x}')
                selected[i].append(f_copy[i][x])
                print(f'вибрані:\n{selected}')
                # for j in range(m):
                #     f_copy[j][x] = 0
                print(f'видалення кандидата:')
                f_copy[:, x] = 0
                for row in f_copy:
                    print(row)
                v_copy[i] -= 1
                print(f'оновлена квота v[{i}] = {v_copy[i]}')
                print()
            else:
                p_copy[i] = 0
        else:
            p_copy[i] = 0
        if np.max(p_copy) == 0:
            z = 0
            real_z = 0
            for i in range(n):
                z += round(np.sum(selected[i]) * p[i], 2)
                real_z += round(np.sum(selected[i]), 2)
            break

    # Знайти максимальне значення в кожному стовпчику
    ideal_z = 0
    for j in range(m):
        x = max(row[j] for row in f)
        ideal_z += x
        ideal_z = round(ideal_z, 2)
    y = round(real_z / ideal_z, 2)

    print(f'Розподіл:\n{selected}')
    print(f'z = {z}')
    print(f'y = {y}')

    return selected, z, y

def read_individual_task(filename):
    with open(filename, 'rb') as f_in:
        m = np.load(f_in)
        n = np.load(f_in)
        f_matrix = np.load(f_in, allow_pickle=True)
        v_list = np.load(f_in, allow_pickle=True).tolist()
        p_list = np.load(f_in, allow_pickle=True).tolist()
    return m, n, f_matrix, v_list, p_list

m, n, f_matrix, v_list, p_list = read_individual_task('task_m_3_n_3_f_mean_0.5_f_var_0.25_v_mean_3_v_var_0.5_p_mean_0.5_p_var_0.25.txt')

f_rez, z, y = solve_greedy(m, n, f_matrix, v_list, p_list)