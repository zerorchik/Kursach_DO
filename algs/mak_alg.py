# -*- coding: utf-8 -*-

import numpy as np
import copy

def solve_mak(m, n, f, v, p):
    # Допоміжні змінні
    f_copy = copy.deepcopy(f)
    v_copy = copy.deepcopy(v)

    is_ok = True
    selected_a = [[] for _ in range(n)]
    selected_a_index = [[] for _ in range(n)]
    pretended = []
    difference = [[] for _ in range(n)]
    pretended_to_remove = [[] for _ in range(n)]
    x_index = []
    skip = False

    # Вхідна задача
    for i in range(len(v)):
        print(v[i], '\t', p[i], '\t', f_copy[i])
    selected = [[] for _ in range(n)]
    print()

    counter = 1
    print(f'Крок {counter}')
    # # Знайти максимальне значення в кожному стовпчику
    # for i in range(m):
    #     x = max(f_copy[j][i] for j in range(n))
    #     for j in range(n):
    #         if f_copy[j][i] == x:
    #             selected[j].append(x)
    #             f_copy[j][i] = 0
    # # Знайти максимальне значення в кожному стовпчику
    # for i in range(m):
    #     max_values = []
    #     for j in range(n):
    #         if f_copy[j][i] > 0:
    #             max_values.append(f_copy[j][i])
    #     max_values.sort(reverse=True)
    #     if len(max_values) > 1 and max_values[0] == max_values[1]:
    #         selected[max_values.index(max_values[0])].append(max_values[0])
    #         f_copy[max_values.index(max_values[0])][i] = 0
    #     else:
    #         for j in range(n):
    #             if f_copy[j][i] == max_values[0]:
    #                 selected[j].append(max_values[0])
    #                 f_copy[j][i] = 0
    # Знайти максимальне значення в кожному стовпчику
    for j in range(m):
        max_elem = -float('inf')
        max_index = -1
        for i in range(n):
            if f[i][j] > max_elem:
                max_elem = f[i][j]
                max_index = i
            elif f[i][j] == max_elem:
                if p[i] > p[max_index]:
                    max_index = i
        selected[max_index].append(max_elem)
    print(f'максимальні значення для кандидатів:\n{selected}')
    print(f'f_copy:')
    for row in f_copy:
        print(row)

    for i in range(n):
        print(f'квота для вакансії [{i}] = {v_copy[i]}, призначено = {len(selected[i])}')
        # Якщо недопустимий розв'язок
        if len(selected[i]) > v[i]:
            print('розклад не допустимий')
            is_ok = False
            # selected_a = [[] for _ in range(n)]
            # selected_a_index = [[] for _ in range(n)]
            # difference = [[] for _ in range(n)]
            # pretended = []
            # x_index = []
            # pretended_to_remove = [[] for _ in range(n)]
            counter += 1
            print(f'\nКрок {counter}')
            for j in range(m):
                for k in range(len(selected[i])):
                    if f[i][j] == selected[i][k]:
                        print(f'призначення, які треба поміняти: {f[i][j]}')
                        selected_a[i].append(j)
                        selected_a_index[i].append(i)
            # Знайти максимальне значення в кожному стовпчику
            for candidate in selected_a[i]:
                max_value = float('-inf')
                x = -1
                index = 0
                for vacancy in range(n):
                    current_value = f_copy[vacancy][candidate]
                    if current_value > max_value:
                        max_value = current_value
                        x = f_copy[vacancy][candidate]
                        index = vacancy
                if x != 0:
                    print(f'кандидат на заміну {f[i][candidate]} - {x}')
                pretended.append(x)
                pretended_to_remove[i].append(f[i][candidate])
                if len(pretended) == len(selected[i]) and all(x == 0 for x in pretended):
                    min = np.min(selected[i])
                    print('немає на кого міняти')
                    print(f'скоротимо найменш ефективне призначення {min}')
                    selected[i].remove(min)
                    print(selected)
                    skip = True
                    break
                diff = round(f[i][candidate] - x, 2)
                if x == 0:
                    diff = 99999
                difference[i].append(diff)
                x_index.append(index)

            if skip:
                print()
                break
            print(f'різниця:\n{difference[i]}')
            x = np.min(difference[i])
            index = np.argmin(difference[i])
            print(f'мінімальна різниця = {x}')
            print(f'прибираємо {pretended_to_remove[i][index]} з призначень:')
            selected[i].remove(pretended_to_remove[i][index])
            print(selected)
            print(f'додаємо {f_copy[x_index[index]][selected_a[i][index]]} до призначень')
            selected[x_index[index]].append(f_copy[x_index[index]][selected_a[i][index]])
            selected[x_index[index]] = sorted(selected[x_index[index]], key=lambda value: f[x_index[index]].index(value))
            f_copy[x_index[index]][selected_a[i][index]] = 0
            x_index = []
            print(selected)
            print(f'f_copy')
            for row in f_copy:
                print(row)
            print()
            break

    while not is_ok:
        selected_a = [[] for _ in range(n)]
        selected_a_index = [[] for _ in range(n)]
        difference = [[] for _ in range(n)]
        pretended = []
        x_index = []
        pretended_to_remove = [[] for _ in range(n)]
        skip = False
        counter += 1
        print(f'Крок {counter}')
        for i in range(n):
            print(f'квота для вакансії [{i}] = {v_copy[i]}, призначено = {len(selected[i])}')
            # Якщо недопустимий розв'язок
            if len(selected[i]) > v[i]:
                print('розклад не допустимий')
                is_ok = False
                counter += 1
                print(f'\nКрок {counter}')
                for j in range(m):
                    for k in range(len(selected[i])):
                        if f[i][j] == selected[i][k]:
                            print(f'призначення, які треба поміняти: {f[i][j]}')
                            selected_a[i].append(j)
                            selected_a_index[i].append(i)
                # Знайти максимальне значення в кожному стовпчику
                for candidate in selected_a[i]:
                    max_value = float('-inf')
                    x = -1
                    index = 0
                    for vacancy in range(n):
                        current_value = f_copy[vacancy][candidate]
                        if current_value > max_value:
                            max_value = current_value
                            x = f_copy[vacancy][candidate]
                            index = vacancy
                    if x != 0:
                        print(f'кандидат на заміну {f[i][candidate]} - {x}')
                    pretended.append(x)
                    pretended_to_remove[i].append(f[i][candidate])
                    if len(pretended) == len(selected[i]) and all(x == 0 for x in pretended):
                        min = np.min(selected[i])
                        print('немає на кого міняти')
                        print(f'скоротимо найменш ефективне призначення {min}')
                        selected[i].remove(min)
                        print(selected)
                        skip = True
                        break
                    diff = round(f[i][candidate] - x, 2)
                    if x == 0:
                        diff = 99999
                    if diff < 0:
                        diff = abs(diff)
                    difference[i].append(diff)
                    x_index.append(index)

                if skip:
                    print()
                    break
                print(f'різниця:\n{difference[i]}')
                x = np.min(difference[i])
                index = np.argmin(difference[i])
                print(f'мінімальна різниця = {x}')
                print(f'прибираємо {pretended_to_remove[i][index]} з призначень:')
                selected[i].remove(pretended_to_remove[i][index])
                print(selected)
                print(f'додаємо {f_copy[x_index[index]][selected_a[i][index]]} до призначень')
                selected[x_index[index]].append(f_copy[x_index[index]][selected_a[i][index]])
                selected[x_index[index]] = sorted(selected[x_index[index]],
                                                  key=lambda value: f[x_index[index]].index(value))
                f_copy[x_index[index]][selected_a[i][index]] = 0
                x_index = []
                print(selected)
                print(f'f_copy')
                for row in f_copy:
                    print(row)
                print()
                break
            # Якщо допустимий
            else:
                is_ok = True

    # Розрахунок ЦФ
    if is_ok:
        print('розклад допустимий\n')
        z = 0
        real_z = 0
        for i in range(n):
            z += round(np.sum(selected[i]) * p[i], 2)
            real_z += round(np.sum(selected[i]), 2)
    z = round(z, 2)
    # Розрахунок ЦФ 2
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
        f_matrix = np.load(f_in, allow_pickle=True).tolist()
        v_list = np.load(f_in, allow_pickle=True).tolist()
        p_list = np.load(f_in, allow_pickle=True).tolist()
    return m, n, f_matrix, v_list, p_list


m, n, f_matrix, v_list, p_list = read_individual_task('task_m_3_n_3_f_mean_0.5_f_var_0.25_v_mean_3_v_var_0.5_p_mean_0.5_p_var_0.25.txt')
# # Захардкоджені параметри
# m = 5
# f_matrix = [[0.63, 0.74, 0.1, 0, 0], [0, 0.7, 0.2, 0, 0.81], [0, 0, 0.9, 0.8, 0.39]]
# v_list = [3, 5, 2]
# p_list = [0.8, 0.5, 0.4]

n = 3
m = 3
f_matrix = [[0.51, 0.35, 0.34], [0.28, 0.61, 0.62], [0.51, 0.53, 0.46]]
v_list = [3, 3, 3]
p_list = [0.21, 0.52, 0.41]

f_rez, z, y = solve_mak(m, n, f_matrix, v_list, p_list)