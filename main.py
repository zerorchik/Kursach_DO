# -*- coding: utf-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt
import time

def generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var):
    # Генерація матриці ефективності призначення кандидатів на вакансії
    f = np.random.normal(f_mean, f_var, (n, m))
    f = np.maximum(0, np.minimum(1, np.round(f, 2)))

    # Генерація квот вакансій
    v = np.random.normal(v_mean, v_var, n)
    v = np.maximum(1, np.round(v)).astype(int)

    # Генерація квот платні
    p = np.random.normal(p_mean, p_var, n)
    p = np.maximum(0.01, np.minimum(1, np.round(p, 2)))

    return f, v, p

def solve_greedy_print(m, n, f, v, p):
    f_copy = copy.deepcopy(f)
    v_copy = copy.deepcopy(v)
    p_copy = copy.deepcopy(p)

    # # Вхідна задача
    # for i in range(len(v)):
    #     print(v[i], '\t', p[i], '\t', f_copy[i])
    # selected = [[] for _ in range(n)]
    # print()

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
    z = round(z, 2)

    # print(f'Розподіл:\n{selected}')
    # print(f'z = {z}')
    # print(f'y = {y}')

    return selected, z, y

def solve_mak_print(m, n, f, v, p):
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

    # # Вхідна задача
    # for i in range(len(v)):
    #     print(v[i], '\t', p[i], '\t', f_copy[i])
    # selected = [[] for _ in range(n)]
    # print()

    counter = 1
    print(f'Крок {counter}')
    # Знайти максимальне значення в кожному стовпчику
    for i in range(m):
        x = max(f_copy[j][i] for j in range(n))
        for j in range(n):
            if f_copy[j][i] == x:
                selected[j].append(x)
                f_copy[j][i] = 0
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

    # print(f'Розподіл:\n{selected}')
    # print(f'z = {z}')
    # print(f'y = {y}')

    return selected, z, y

def solve_hungarian_print(m, n, f, v, p):
    f_copy = copy.deepcopy(f)
    selected = [[] for _ in range(n)]
    max_f = [[] for _ in range(n)]
    max_f_counts = [0] * len(f_copy[0])
    stop = False

    # # Вхідна задача
    # for i in range(len(v)):
    #     print(v[i], '\t', p[i], '\t', f_copy[i])
    # selected = [[] for _ in range(n)]
    # print()

    for i in range(n):
        for j in range(m):
            if f_copy[i][j] == 0:
                f_copy[i][j] = -np.inf
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
                max_f[i].append(max_val)
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
                    max_f[i].append(max_val)
                    print(f'max_f[{i}] = {max_f[i][0]}')
                    max_f_counts[max_index] += 1

        for i in range(n):
            if len(max_f[i]) > 0:
                for j in range(m):
                    if f_copy[i][j] != -np.inf and f_copy[i][j] != 0:
                        f_copy[i][j] = round(max_f[i][0] - f_copy[i][j], 2)
                print(f'{max_f[i][0]} - f_copy[{i}] = {f_copy[i]}')

        for i in range(n):
            for j in range(m):
                if f_copy[i][j] == 0 and max_f_counts[j] == 1:
                    print(f'{f[i][j]} - єдиний макс елемент у стовпці')
                    if v[i] > 0:
                        print('квота виконується')
                        selected[i].append(j)
                        print(f'selected:\n{selected}')
                        v[i] -= 1  # update quota
                        print(f'нова квота\n{v}')
                        print('кандидата назначено')
                        f_copy[:, j] = -np.inf
                        print(f_copy)
            if v[i] == 0:
                print('вакансію закрито')
                f_copy[i, :] = -np.inf
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
    print()

    # print(f'\nРозподіл:\n{selected_values}')
    # print(f'z = {z}')
    # print(f'y = {y}')

    return selected_values, z, y

def solve_greedy(m, n, f, v, p):
    f_copy = copy.deepcopy(f)
    v_copy = copy.deepcopy(v)
    p_copy = copy.deepcopy(p)

    # Вхідна задача
    selected = [[] for _ in range(n)]

    for step in range(n * m):
        # Найбільший пріоритет
        i = np.argmax(p_copy)
        if len(selected[i]) < v[i]:
            x = np.argmax(f_copy[i])
            if f_copy[i][x] != 0:
                selected[i].append(f_copy[i][x])
                f_copy[:, x] = 0
                v_copy[i] -= 1
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
    z = round(z, 2)

    return selected, z, y

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
    selected = [[] for _ in range(n)]

    counter = 1
    # Знайти максимальне значення в кожному стовпчику
    # for i in range(m):
    #     x = max(f_copy[j][i] for j in range(n))
    #     for j in range(n):
    #         if f_copy[j][i] == x:
    #             selected[j].append(x)
    #             f_copy[j][i] = 0
    # Знаходження максимального елемента в кожному стовпчику матриці f
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
    # Знаходження максимального елемента в кожному стовпчику матриці f
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

    for i in range(n):
        # Якщо недопустимий розв'язок
        if len(selected[i]) > v[i]:
            is_ok = False
            counter += 1
            for j in range(m):
                for k in range(len(selected[i])):
                    if f[i][j] == selected[i][k]:
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
                pretended.append(x)
                pretended_to_remove[i].append(f[i][candidate])
                if len(pretended) == len(selected[i]) and all(x == 0 for x in pretended):
                    min = np.min(selected[i])
                    selected[i].remove(min)
                    skip = True
                    break
                diff = round(f[i][candidate] - x, 2)
                if x == 0:
                    diff = 99999
                difference[i].append(diff)
                x_index.append(index)

            if skip:
                break
            x = np.min(difference[i])
            index = np.argmin(difference[i])
            selected[i].remove(pretended_to_remove[i][index])
            selected[x_index[index]].append(f_copy[x_index[index]][selected_a[i][index]])
            selected[x_index[index]] = sorted(selected[x_index[index]], key=lambda value: f[x_index[index]].index(value))
            f_copy[x_index[index]][selected_a[i][index]] = 0
            x_index = []
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
        for i in range(n):
            # Якщо недопустимий розв'язок
            if len(selected[i]) > v[i]:
                is_ok = False
                counter += 1
                for j in range(m):
                    for k in range(len(selected[i])):
                        if f[i][j] == selected[i][k]:
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
                    pretended.append(x)
                    pretended_to_remove[i].append(f[i][candidate])
                    if len(pretended) == len(selected[i]) and all(x == 0 for x in pretended):
                        min = np.min(selected[i])
                        selected[i].remove(min)
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
                    break
                x = np.min(difference[i])
                index = np.argmin(difference[i])
                selected[i].remove(pretended_to_remove[i][index])
                selected[x_index[index]].append(f_copy[x_index[index]][selected_a[i][index]])
                selected[x_index[index]] = sorted(selected[x_index[index]],
                                                  key=lambda value: f[x_index[index]].index(value))
                f_copy[x_index[index]][selected_a[i][index]] = 0
                x_index = []
                break
            # Якщо допустимий
            else:
                is_ok = True

    # Розрахунок ЦФ
    if is_ok:
        z = 0
        real_z = 0
        for i in range(n):
            z += round(np.sum(selected[i]) * p[i], 2)
            real_z += round(np.sum(selected[i]), 2)
    z = round(z, 2)
    real_z = round(real_z, 2)
    # Розрахунок ЦФ 2
    ideal_z = 0
    # for j in range(m):
    #     # x = max(row[j] for row in f)
    #     # x = max(f[i][j] for i in range(n))
    #     ideal_z += x
    #     ideal_z = round(ideal_z, 2)
    for j in range(len(f[0])):
        max_column = -1
        for row in f:
            if row[j] > max_column:
                max_column = row[j]
        ideal_z += max_column
    ideal_z = round(ideal_z, 2)

    y = round(real_z / ideal_z, 2)

    return selected, z, y

def solve_hungarian(m, n, f, v, p):
    f_copy = copy.deepcopy(f)
    selected = [[] for _ in range(n)]
    max_f = [[] for _ in range(n)]
    max_f_counts = [0] * len(f_copy[0])
    stop = False

    # Вхідна задача
    selected = [[] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if f_copy[i][j] == 0:
                f_copy[i][j] = -np.inf

    counter = 0
    while not stop:
        counter += 1
        # Якщо на мінімум
        if counter > 1:
            for i in range(n):
                for j in range(m):
                    if f_copy[i][j] != -np.inf and f_copy[i][j] != 0:
                        f_copy[i][j] = 1 - f_copy[i][j]

        all_is_null = False
        for i in range(n):
            max_val = round(np.max(f_copy[i]), 2)
            max_index = np.argmax(f_copy[i])
            if max_val != -np.inf:
                max_f[i].append(max_val)
                max_f_counts[max_index] += 1
        if all(value == 0 for sublist in max_f for value in sublist):
            all_is_null = True

        if all_is_null:
            max_f = [[] for _ in range(n)]
            for j in range(m):
                if max_f_counts[j] == n:
                    max_f_counts = [0] * len(f_copy[0])
                    max_val = round(np.max(f[j]), 2)
                    max_f[i].append(max_val)
                    max_f_counts[max_index] += 1

        for i in range(n):
            if len(max_f[i]) > 0:
                for j in range(m):
                    if f_copy[i][j] != -np.inf and f_copy[i][j] != 0:
                        f_copy[i][j] = round(max_f[i][0] - f_copy[i][j], 2)

        for i in range(n):
            for j in range(m):
                if f_copy[i][j] == 0 and max_f_counts[j] == 1:
                    if v[i] > 0:
                        selected[i].append(j)
                        v[i] -= 1
                        f_copy[:, j] = -np.inf
            if v[i] == 0:
                f_copy[i, :] = -np.inf

        max_f = [[] for _ in range(n)]
        max_f_counts = [0] * len(f_copy[0])

        if np.all(v == 0) or np.all(f_copy == -np.inf):
            stop = True

        if counter == 100 * max(n, m):
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

    ideal_z = 0
    for j in range(m):
        x = max(row[j] for row in f)
        ideal_z += x
        ideal_z = round(ideal_z, 2)
    y = round(real_z / ideal_z, 2)
    z = round(z, 2)

    return selected_values, z, y

# ----------------------------------------------------------------------------------------------------------------------
'''------------------------------------------------Головні виклики---------------------------------------------------'''
# ----------------------------------------------------------------------------------------------------------------------

print('\nГоловне меню.\n')
print('Доступні опції:')
options_menu = ['згенерувати задачу',
                'приклади роботи алгоритмів',
                'експерименти',
                'розв\'язати задачу всіма алгоритмами']
for i in range(len(options_menu)):
    print(f'{i + 1}. {options_menu[i]}')
data_var = int(input('mode:'))

# Генерація
if data_var == 1:
    print('')
    print('-' * 50)
    print('\nГенерація задачі.\n')
    print('Доступні опції:')
    options_gen = ['параметри за замовчуванням',
                   'задати параметри вручну']
    for i in range(len(options_gen)):
        print(f'{i + 1}. {options_gen[i]}')
    data_var_gen = int(input('mode:'))

    # Автоматично
    if data_var_gen == 1:
        print('\nАвтоматична генерація.')
        m = 3
        n = 3
        f_mean = 0.5
        f_var = 0.25
        v_mean = 3
        v_var = 0.5
        p_mean = 0.5
        p_var = 0.25

        f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

    # Вручну
    if data_var_gen == 2:
        print()
        print('-' * 50)
        print('\nВизначення параметрів.\n')
        m = int(input('Кількість кандидатів (3-100)\nm = '))
        n = int(input('Кількість вакансій (3-14)\nn = '))

        f_mean = float(input('Мат. спод. еф. призн. f (0.1-1)\nf_mean = '))
        f_var = float(input('Дисп. еф. призн. f (0.1-1)\nf_var = '))

        v_mean = float(input('Мат. спод. квот v (1-20)\nv_mean = '))
        v_var = float(input('Дисп. квот v (0.1-1)\nv_var = '))

        p_mean = float(input('Мат. спод. пріорит. p (0.1-1)\np_mean = '))
        p_var = float(input('Дисп. пріорит. p (0.1-1)\np_var = '))

        f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

    # Вхідна задача
    print()
    print('-' * 50)
    print('Результат')
    print('-' * 50)
    print('Вхідна задача:')
    print(f'Кандидатів = {m}')
    print(f'Вакансій = {n}')
    print()
    print('v[i]\tp[i]\tf[i][j]')
    for i in range(len(v)):
        print(v[i], '\t\t', p[i], '\t', f[i])
    selected = [[] for _ in range(n)]
    print()

# Приклади
if data_var == 2:
    print('')
    print('-' * 50)
    print('\nПриклади роботи алгоритмів.\n')
    print('Доступні опції:')
    options_ex = ['Жадібний алгоритм',
                  'алгоритм Мака',
                  'Угорський алгоритм']
    for i in range(len(options_ex)):
        print(f'{i + 1}. {options_ex[i]}')
    data_var_ex = int(input('mode:'))

    # Жадібний
    if data_var_ex == 1:
        print('')
        print('-' * 50)
        print('\nЖадібний алгоритм.\n')
        print('Доступні опції:')
        options_jeal = ['задача 1 (Павлова)',
                        'задача 2 (Гоголь)']
        for i in range(len(options_jeal)):
            print(f'{i + 1}. {options_jeal[i]}')
        data_var_jeal = int(input('mode:'))

        # Павлова
        if data_var_jeal == 1:
            num = 1
            name = 'Павлова'
            m = 5
            n = 3
            f = np.array(
                [[0.96, 0.91, 0.8, 0, 0.97],
                 [0, 0.7, 0.51, 0.72, 0],
                 [0.5, 0, 0.6, 0.6, 0.99]])
            v = [2, 1, 2]
            p = [0.5, 0.3, 0.2]

        # Гоголь
        if data_var_jeal == 2:
            num = 2
            name = 'Гоголь'
            m = 5
            n = 3
            f = np.array(
                [[0.63, 0.74, 0.1, 0, 0],
                 [0, 0.7, 0.2, 0, 0.81],
                 [0, 0, 0.9, 0.8, 0.39]])
            v = [3, 5, 2]
            p = [0.8, 0.5, 0.4]

        # Вхідна задача
        print()
        print('-' * 50)
        print(f'Задача {num} ({name})')
        print('-' * 50)
        print(f'Кандидатів = {m}')
        print(f'Вакансій = {n}')
        print()
        print('v[i]\tp[i]\tf[i][j]')
        for i in range(len(v)):
            print(v[i], '\t\t', p[i], '\t', f[i])
        selected = [[] for _ in range(n)]
        print()

        # Розв'язок
        print('-' * 50)
        print('Розв\'язання')
        print('-' * 50)
        f_rez, z, y = solve_greedy_print(m, n, f, v, p)

        # Результат
        print('-' * 50)
        print('Розподіл')
        print('-' * 50)
        print(f'{f_rez}')
        print(f'z = {z}')
        print(f'y = {y}')

    # Мака
    if data_var_ex == 2:
        print('')
        print('-' * 50)
        print('\nАлгоритм Мака.\n')
        print('Доступні опції:')
        options_jeal = ['задача 1 (Павлова)',
                        'задача 2 (Гоголь)']
        for i in range(len(options_jeal)):
            print(f'{i + 1}. {options_jeal[i]}')
        data_var_jeal = int(input('mode:'))

        # Павлова
        if data_var_jeal == 1:
            num = 1
            name = 'Павлова'
            m = 5
            n = 3
            f = np.array(
                [[0.96, 0.91, 0.8, 0, 0.97],
                 [0, 0.7, 0.51, 0.72, 0],
                 [0.5, 0, 0.6, 0.6, 0.99]]).tolist()
            v = [2, 1, 2]
            p = [0.5, 0.3, 0.2]

        # Гоголь
        if data_var_jeal == 2:
            num = 2
            name = 'Гоголь'
            m = 5
            n = 3
            f = np.array(
                [[0.63, 0.74, 0.1, 0, 0],
                 [0, 0.7, 0.2, 0, 0.81],
                 [0, 0, 0.9, 0.8, 0.39]]).tolist()
            v = [3, 5, 2]
            p = [0.8, 0.5, 0.4]

        # Вхідна задача
        print()
        print('-' * 50)
        print(f'Задача {num} ({name})')
        print('-' * 50)
        print(f'Кандидатів = {m}')
        print(f'Вакансій = {n}')
        print()
        print('v[i]\tp[i]\tf[i][j]')
        for i in range(len(v)):
            print(v[i], '\t\t', p[i], '\t', f[i])
        selected = [[] for _ in range(n)]
        print()

        # Розв'язок
        print('-' * 50)
        print('Розв\'язання')
        print('-' * 50)
        f_rez, z, y = solve_mak_print(m, n, f, v, p)

        # Результат
        print('-' * 50)
        print('Розподіл')
        print('-' * 50)
        print(f'{f_rez}')
        print(f'z = {z}')
        print(f'y = {y}')

    # Угорський
    if data_var_ex == 3:
        print('')
        print('-' * 50)
        print('\nУгорський алгоритм.\n')
        print('Доступні опції:')
        options_jeal = ['задача 1 (Павлова)',
                        'задача 2 (Гоголь)']
        for i in range(len(options_jeal)):
            print(f'{i + 1}. {options_jeal[i]}')
        data_var_jeal = int(input('mode:'))

        # Павлова
        if data_var_jeal == 1:
            num = 1
            name = 'Павлова'
            m = 5
            n = 3
            f = np.array(
                [[0.96, 0.91, 0.8, 0, 0.97],
                 [0, 0.7, 0.51, 0.72, 0],
                 [0.5, 0, 0.6, 0.6, 0.99]]).tolist()
            v = [2, 1, 2]
            p = [0.5, 0.3, 0.2]

        # Гоголь
        if data_var_jeal == 2:
            num = 2
            name = 'Гоголь'
            m = 5
            n = 3
            f = np.array(
                [[0.63, 0.74, 0.1, 0, 0],
                 [0, 0.7, 0.2, 0, 0.81],
                 [0, 0, 0.9, 0.8, 0.39]])
            v = [3, 5, 2]
            p = [0.8, 0.5, 0.4]

        # Вхідна задача
        print()
        print('-' * 50)
        print(f'Задача {num} ({name})')
        print('-' * 50)
        print(f'Кандидатів = {m}')
        print(f'Вакансій = {n}')
        print()
        print('v[i]\tp[i]\tf[i][j]')
        for i in range(len(v)):
            print(v[i], '\t\t', p[i], '\t', f[i])
        selected = [[] for _ in range(n)]
        print()

        # Розв'язок
        print('-' * 50)
        print('Розв\'язання')
        print('-' * 50)
        f_rez, z, y = solve_hungarian_print(m, n, f, v, p)

        # Результат
        print('-' * 50)
        print('Розподіл')
        print('-' * 50)
        print(f'{f_rez}')
        print(f'z = {z}')
        print(f'y = {y}')

# Приклади
if data_var == 3:
    print('')
    print('-' * 50)
    print('\nЕксперименти.\n')

    # Список значень параметра m
    m_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    f_means = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    f_vars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    v_means = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    v_vars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    p_means = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    p_vars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Список для зберігання середнього часу роботи кожного алгоритму
    avg_times_greedy = []
    avg_times_mak = []
    avg_times_hungarian = []

    print('Виберіть експеримент:')
    experiments = ['розмірність / час - m (вакансії)',
                   'розмірність / час - n (кандидати)',
                   'розмірність / точність - m (вакансії)',
                   'розмірність / точність - n (кандидати)',
                   'параметри / точність - f_mean (мат. спод. еф. призн.)',
                   'параметри / точність - v_mean (мат. спод. квот)',
                   'параметри / точність - p_mean (мат. спод. пріорит.)',
                   'параметри / точність - f_var (дисп. еф. призн.)',
                   'параметри / точність - v_var (дисп. квот)',
                   'параметри / точність - p_var (дисп. пріорит.)', ]
    for i in range(len(experiments)):
        print(f'{i + 1}. {experiments[i]}')

    print('\nДля виходу уведіть "0"\n')

    while True:
        data_var_exp = int(input('mode:'))

        if data_var_exp == 0:
            break
        # Проведення 20 експериментів для кожного значення параметра m
        if data_var_exp == 1:
            for m in m_values:
                times_greedy = []
                times_mak = []
                times_hungarian = []

                for _ in range(20):
                    # Генерація задачі
                    n = 3
                    f_mean = 0.5
                    f_var = 0.25
                    v_mean = 3
                    v_var = 0.5
                    p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    start_time_greedy = time.time()
                    f_rez, z, y = solve_greedy(m, n, f, v, p)
                    end_time_greedy = time.time()
                    times_greedy.append(end_time_greedy - start_time_greedy)

                    f_list = f.tolist()
                    start_time_mak = time.time()
                    f_rez, z, y = solve_mak(m, n, f_list, v, p)
                    end_time_mak = time.time()
                    times_mak.append(end_time_mak - start_time_mak)

                    start_time_hungarian = time.time()
                    f_rez, z, y = solve_hungarian(m, n, f, v, p)
                    end_time_hungarian = time.time()
                    times_hungarian.append(end_time_hungarian - start_time_hungarian)

                # Розрахунок середнього часу роботи кожного алгоритму
                avg_time_greedy = np.mean(times_greedy)
                avg_time_mak = np.mean(times_mak)
                avg_time_hungarian = np.mean(times_hungarian)

                avg_times_greedy.append(avg_time_greedy)
                avg_times_mak.append(avg_time_mak)
                avg_times_hungarian.append(avg_time_hungarian)

            # Візуалізація результатів
            plt.plot(m_values, avg_times_greedy, label='Greedy')
            plt.plot(m_values, avg_times_mak, label='Mak')
            plt.plot(m_values, avg_times_hungarian, label='Hungarian')
            plt.xlabel('m')
            plt.ylabel('Average time (s)')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра n
        if data_var_exp == 2:
            for n in n_values:
                times_greedy = []
                times_mak = []
                times_hungarian = []

                for _ in range(20):
                    # Генерація задачі
                    m = 3
                    f_mean = 0.5
                    f_var = 0.25
                    v_mean = 3
                    v_var = 0.5
                    p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    start_time_greedy = time.time()
                    f_rez, z, y = solve_greedy(m, n, f, v, p)
                    end_time_greedy = time.time()
                    times_greedy.append(end_time_greedy - start_time_greedy)

                    f_list = f.tolist()
                    start_time_mak = time.time()
                    f_rez, z, y = solve_mak(m, n, f_list, v, p)
                    end_time_mak = time.time()
                    times_mak.append(end_time_mak - start_time_mak)

                    start_time_hungarian = time.time()
                    f_rez, z, y = solve_hungarian(m, n, f, v, p)
                    end_time_hungarian = time.time()
                    times_hungarian.append(end_time_hungarian - start_time_hungarian)

                # Розрахунок середнього часу роботи кожного алгоритму
                avg_time_greedy = np.mean(times_greedy)
                avg_time_mak = np.mean(times_mak)
                avg_time_hungarian = np.mean(times_hungarian)

                avg_times_greedy.append(avg_time_greedy)
                avg_times_mak.append(avg_time_mak)
                avg_times_hungarian.append(avg_time_hungarian)

            # Візуалізація результатів
            plt.plot(n_values, avg_times_greedy, label='Greedy')
            plt.plot(n_values, avg_times_mak, label='Mak')
            plt.plot(n_values, avg_times_hungarian, label='Hungarian')
            plt.xlabel('n')
            plt.ylabel('Average time (s)')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра m (точність)
        if data_var_exp == 3:
            avg_z_greedy_val = []
            avg_z_mak_val = []
            avg_z_hungarian_val = []

            avg_y_greedy_val = []
            avg_y_mak_val = []
            avg_y_hungarian_val = []
            for m in m_values:
                # Створіть списки для зберігання середніх значень цільових функцій
                z_greedy_val = []
                z_mak_val = []
                z_hungarian_val = []

                y_greedy_val = []
                y_mak_val = []
                y_hungarian_val = []

                for _ in range(20):
                    # Генерація задачі
                    n = 3
                    f_mean = 0.5
                    f_var = 0.25
                    v_mean = 3
                    v_var = 0.5
                    p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    f_rez, z_greedy, y_greedy = solve_greedy(m, n, f, v, p)
                    z_greedy_val.append(z_greedy)
                    y_greedy_val.append(y_greedy)

                    f_list = f.tolist()
                    f_rez, z_mak, y_mak = solve_mak(m, n, f_list, v, p)
                    z_mak_val.append(z_mak)
                    y_mak_val.append(y_mak)

                    f_rez, z_hungarian, y_hungarian = solve_hungarian(m, n, f, v, p)
                    z_hungarian_val.append(z_hungarian)
                    y_hungarian_val.append(y_hungarian)

                # Розрахуйте середні значення цільових функцій до списків
                avg_z_greedy = np.mean(z_greedy_val)
                avg_z_mak = np.mean(z_mak_val)
                avg_z_hungarian = np.mean(z_hungarian_val)

                avg_y_greedy = np.mean(y_greedy_val)
                avg_y_mak = np.mean(y_mak_val)
                avg_y_hungarian = np.mean(y_hungarian_val)

                # Додайте середні значення цільових функцій до списків
                avg_z_greedy_val.append(avg_z_greedy)
                avg_z_mak_val.append(avg_z_mak)
                avg_z_hungarian_val.append(avg_z_hungarian)

                avg_y_greedy_val.append(avg_y_greedy)
                avg_y_mak_val.append(avg_y_mak)
                avg_y_hungarian_val.append(avg_y_hungarian)

            # Створіть перший графік для середніх значень цільової функції z
            plt.plot(m_values, avg_z_greedy_val, label='Greedy')
            plt.plot(m_values, avg_z_mak_val, label='Mak')
            plt.plot(m_values, avg_z_hungarian_val, label='Hungarian')
            plt.xlabel('m')
            plt.ylabel('Середнє значення цільової функції z')
            plt.title('Середнє значення цільової функції z для різних алгоритмів')
            plt.legend()
            plt.show()

            # Створіть другий графік для середніх значень другої цільової функції y
            plt.plot(m_values, avg_y_greedy_val, label='Greedy')
            plt.plot(m_values, avg_y_mak_val, label='Mak')
            plt.plot(m_values, avg_y_hungarian_val, label='Hungarian')
            plt.xlabel('m')
            plt.ylabel('Середнє значення другої цільової функції y')
            plt.title('Середнє значення другої цільової функції y для різних алгоритмів')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра n (точність)
        if data_var_exp == 4:
            avg_z_greedy_val = []
            avg_z_mak_val = []
            avg_z_hungarian_val = []

            avg_y_greedy_val = []
            avg_y_mak_val = []
            avg_y_hungarian_val = []
            for n in n_values:
                # Створіть списки для зберігання середніх значень цільових функцій
                z_greedy_val = []
                z_mak_val = []
                z_hungarian_val = []

                y_greedy_val = []
                y_mak_val = []
                y_hungarian_val = []

                for _ in range(20):
                    # Генерація задачі
                    m = 3
                    f_mean = 0.5
                    f_var = 0.25
                    v_mean = 3
                    v_var = 0.5
                    p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    f_rez, z_greedy, y_greedy = solve_greedy(m, n, f, v, p)
                    z_greedy_val.append(z_greedy)
                    y_greedy_val.append(y_greedy)

                    f_list = f.tolist()
                    f_rez, z_mak, y_mak = solve_mak(m, n, f_list, v, p)
                    z_mak_val.append(z_mak)
                    y_mak_val.append(y_mak)

                    f_rez, z_hungarian, y_hungarian = solve_hungarian(m, n, f, v, p)
                    z_hungarian_val.append(z_hungarian)
                    y_hungarian_val.append(y_hungarian)

                # Розрахуйте середні значення цільових функцій до списків
                avg_z_greedy = np.mean(z_greedy_val)
                avg_z_mak = np.mean(z_mak_val)
                avg_z_hungarian = np.mean(z_hungarian_val)

                avg_y_greedy = np.mean(y_greedy_val)
                avg_y_mak = np.mean(y_mak_val)
                avg_y_hungarian = np.mean(y_hungarian_val)

                # Додайте середні значення цільових функцій до списків
                avg_z_greedy_val.append(avg_z_greedy)
                avg_z_mak_val.append(avg_z_mak)
                avg_z_hungarian_val.append(avg_z_hungarian)

                avg_y_greedy_val.append(avg_y_greedy)
                avg_y_mak_val.append(avg_y_mak)
                avg_y_hungarian_val.append(avg_y_hungarian)

            # Створіть перший графік для середніх значень цільової функції z
            plt.plot(n_values, avg_z_greedy_val, label='Greedy')
            plt.plot(n_values, avg_z_mak_val, label='Mak')
            plt.plot(n_values, avg_z_hungarian_val, label='Hungarian')
            plt.xlabel('n')
            plt.ylabel('Середнє значення цільової функції z')
            plt.title('Середнє значення цільової функції z для різних алгоритмів')
            plt.legend()
            plt.show()

            # Створіть другий графік для середніх значень другої цільової функції y
            plt.plot(n_values, avg_y_greedy_val, label='Greedy')
            plt.plot(n_values, avg_y_mak_val, label='Mak')
            plt.plot(n_values, avg_y_hungarian_val, label='Hungarian')
            plt.xlabel('n')
            plt.ylabel('Середнє значення другої цільової функції y')
            plt.title('Середнє значення другої цільової функції y для різних алгоритмів')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра f_mean (точність)
        if data_var_exp == 5:
            avg_z_greedy_val = []
            avg_z_mak_val = []
            avg_z_hungarian_val = []

            avg_y_greedy_val = []
            avg_y_mak_val = []
            avg_y_hungarian_val = []
            for f_mean in f_means:
                # Створіть списки для зберігання середніх значень цільових функцій
                z_greedy_val = []
                z_mak_val = []
                z_hungarian_val = []

                y_greedy_val = []
                y_mak_val = []
                y_hungarian_val = []

                for _ in range(20):
                    # Генерація задачі
                    n = 3
                    m = 3
                    # f_mean = 0.5
                    f_var = 0.25
                    v_mean = 3
                    v_var = 0.5
                    p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    f_rez, z_greedy, y_greedy = solve_greedy(m, n, f, v, p)
                    z_greedy_val.append(z_greedy)
                    y_greedy_val.append(y_greedy)

                    f_list = f.tolist()
                    f_rez, z_mak, y_mak = solve_mak(m, n, f_list, v, p)
                    z_mak_val.append(z_mak)
                    y_mak_val.append(y_mak)

                    f_rez, z_hungarian, y_hungarian = solve_hungarian(m, n, f, v, p)
                    z_hungarian_val.append(z_hungarian)
                    y_hungarian_val.append(y_hungarian)

                # Розрахуйте середні значення цільових функцій до списків
                avg_z_greedy = np.mean(z_greedy_val)
                avg_z_mak = np.mean(z_mak_val)
                avg_z_hungarian = np.mean(z_hungarian_val)

                avg_y_greedy = np.mean(y_greedy_val)
                avg_y_mak = np.mean(y_mak_val)
                avg_y_hungarian = np.mean(y_hungarian_val)

                # Додайте середні значення цільових функцій до списків
                avg_z_greedy_val.append(avg_z_greedy)
                avg_z_mak_val.append(avg_z_mak)
                avg_z_hungarian_val.append(avg_z_hungarian)

                avg_y_greedy_val.append(avg_y_greedy)
                avg_y_mak_val.append(avg_y_mak)
                avg_y_hungarian_val.append(avg_y_hungarian)

            # Створіть перший графік для середніх значень цільової функції z
            plt.plot(f_means, avg_z_greedy_val, label='Greedy')
            plt.plot(f_means, avg_z_mak_val, label='Mak')
            plt.plot(f_means, avg_z_hungarian_val, label='Hungarian')
            plt.xlabel('f_mean')
            plt.ylabel('Середнє значення цільової функції z')
            plt.title('Середнє значення цільової функції z для різних алгоритмів')
            plt.legend()
            plt.show()

            # Створіть другий графік для середніх значень другої цільової функції y
            plt.plot(f_means, avg_y_greedy_val, label='Greedy')
            plt.plot(f_means, avg_y_mak_val, label='Mak')
            plt.plot(f_means, avg_y_hungarian_val, label='Hungarian')
            plt.xlabel('f_mean')
            plt.ylabel('Середнє значення другої цільової функції y')
            plt.title('Середнє значення другої цільової функції y для різних алгоритмів')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра v_mean (точність)
        if data_var_exp == 6:
            avg_z_greedy_val = []
            avg_z_mak_val = []
            avg_z_hungarian_val = []

            avg_y_greedy_val = []
            avg_y_mak_val = []
            avg_y_hungarian_val = []
            for v_mean in v_means:
                # Створіть списки для зберігання середніх значень цільових функцій
                z_greedy_val = []
                z_mak_val = []
                z_hungarian_val = []

                y_greedy_val = []
                y_mak_val = []
                y_hungarian_val = []

                for _ in range(20):
                    # Генерація задачі
                    n = 3
                    m = 3
                    f_mean = 0.5
                    f_var = 0.25
                    # v_mean = 3
                    v_var = 0.5
                    p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    f_rez, z_greedy, y_greedy = solve_greedy(m, n, f, v, p)
                    z_greedy_val.append(z_greedy)
                    y_greedy_val.append(y_greedy)

                    f_list = f.tolist()
                    f_rez, z_mak, y_mak = solve_mak(m, n, f_list, v, p)
                    z_mak_val.append(z_mak)
                    y_mak_val.append(y_mak)

                    f_rez, z_hungarian, y_hungarian = solve_hungarian(m, n, f, v, p)
                    z_hungarian_val.append(z_hungarian)
                    y_hungarian_val.append(y_hungarian)

                # Розрахуйте середні значення цільових функцій до списків
                avg_z_greedy = np.mean(z_greedy_val)
                avg_z_mak = np.mean(z_mak_val)
                avg_z_hungarian = np.mean(z_hungarian_val)

                avg_y_greedy = np.mean(y_greedy_val)
                avg_y_mak = np.mean(y_mak_val)
                avg_y_hungarian = np.mean(y_hungarian_val)

                # Додайте середні значення цільових функцій до списків
                avg_z_greedy_val.append(avg_z_greedy)
                avg_z_mak_val.append(avg_z_mak)
                avg_z_hungarian_val.append(avg_z_hungarian)

                avg_y_greedy_val.append(avg_y_greedy)
                avg_y_mak_val.append(avg_y_mak)
                avg_y_hungarian_val.append(avg_y_hungarian)

            # Створіть перший графік для середніх значень цільової функції z
            plt.plot(v_means, avg_z_greedy_val, label='Greedy')
            plt.plot(v_means, avg_z_mak_val, label='Mak')
            plt.plot(v_means, avg_z_hungarian_val, label='Hungarian')
            plt.xlabel('v_mean')
            plt.ylabel('Середнє значення цільової функції z')
            plt.title('Середнє значення цільової функції z для різних алгоритмів')
            plt.legend()
            plt.show()

            # Створіть другий графік для середніх значень другої цільової функції y
            plt.plot(v_means, avg_y_greedy_val, label='Greedy')
            plt.plot(v_means, avg_y_mak_val, label='Mak')
            plt.plot(v_means, avg_y_hungarian_val, label='Hungarian')
            plt.xlabel('v_mean')
            plt.ylabel('Середнє значення другої цільової функції y')
            plt.title('Середнє значення другої цільової функції y для різних алгоритмів')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра p_mean (точність)
        if data_var_exp == 7:
            avg_z_greedy_val = []
            avg_z_mak_val = []
            avg_z_hungarian_val = []

            avg_y_greedy_val = []
            avg_y_mak_val = []
            avg_y_hungarian_val = []
            for p_mean in p_means:
                # Створіть списки для зберігання середніх значень цільових функцій
                z_greedy_val = []
                z_mak_val = []
                z_hungarian_val = []

                y_greedy_val = []
                y_mak_val = []
                y_hungarian_val = []

                for _ in range(20):
                    # Генерація задачі
                    n = 3
                    m = 3
                    f_mean = 0.5
                    f_var = 0.25
                    v_mean = 3
                    v_var = 0.5
                    # p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    f_rez, z_greedy, y_greedy = solve_greedy(m, n, f, v, p)
                    z_greedy_val.append(z_greedy)
                    y_greedy_val.append(y_greedy)

                    f_list = f.tolist()
                    f_rez, z_mak, y_mak = solve_mak(m, n, f_list, v, p)
                    z_mak_val.append(z_mak)
                    y_mak_val.append(y_mak)

                    f_rez, z_hungarian, y_hungarian = solve_hungarian(m, n, f, v, p)
                    z_hungarian_val.append(z_hungarian)
                    y_hungarian_val.append(y_hungarian)

                # Розрахуйте середні значення цільових функцій до списків
                avg_z_greedy = np.mean(z_greedy_val)
                avg_z_mak = np.mean(z_mak_val)
                avg_z_hungarian = np.mean(z_hungarian_val)

                avg_y_greedy = np.mean(y_greedy_val)
                avg_y_mak = np.mean(y_mak_val)
                avg_y_hungarian = np.mean(y_hungarian_val)

                # Додайте середні значення цільових функцій до списків
                avg_z_greedy_val.append(avg_z_greedy)
                avg_z_mak_val.append(avg_z_mak)
                avg_z_hungarian_val.append(avg_z_hungarian)

                avg_y_greedy_val.append(avg_y_greedy)
                avg_y_mak_val.append(avg_y_mak)
                avg_y_hungarian_val.append(avg_y_hungarian)

            # Створіть перший графік для середніх значень цільової функції z
            plt.plot(p_means, avg_z_greedy_val, label='Greedy')
            plt.plot(p_means, avg_z_mak_val, label='Mak')
            plt.plot(p_means, avg_z_hungarian_val, label='Hungarian')
            plt.xlabel('p_mean')
            plt.ylabel('Середнє значення цільової функції z')
            plt.title('Середнє значення цільової функції z для різних алгоритмів')
            plt.legend()
            plt.show()

            # Створіть другий графік для середніх значень другої цільової функції y
            plt.plot(p_means, avg_y_greedy_val, label='Greedy')
            plt.plot(p_means, avg_y_mak_val, label='Mak')
            plt.plot(p_means, avg_y_hungarian_val, label='Hungarian')
            plt.xlabel('p_mean')
            plt.ylabel('Середнє значення другої цільової функції y')
            plt.title('Середнє значення другої цільової функції y для різних алгоритмів')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра f_var (точність)
        if data_var_exp == 8:
            avg_z_greedy_val = []
            avg_z_mak_val = []
            avg_z_hungarian_val = []

            avg_y_greedy_val = []
            avg_y_mak_val = []
            avg_y_hungarian_val = []
            for f_var in f_vars:
                # Створіть списки для зберігання середніх значень цільових функцій
                z_greedy_val = []
                z_mak_val = []
                z_hungarian_val = []

                y_greedy_val = []
                y_mak_val = []
                y_hungarian_val = []

                for _ in range(20):
                    # Генерація задачі
                    n = 3
                    m = 3
                    f_mean = 0.5
                    # f_var = 0.25
                    v_mean = 3
                    v_var = 0.5
                    p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    f_rez, z_greedy, y_greedy = solve_greedy(m, n, f, v, p)
                    z_greedy_val.append(z_greedy)
                    y_greedy_val.append(y_greedy)

                    f_list = f.tolist()
                    f_rez, z_mak, y_mak = solve_mak(m, n, f_list, v, p)
                    z_mak_val.append(z_mak)
                    y_mak_val.append(y_mak)

                    f_rez, z_hungarian, y_hungarian = solve_hungarian(m, n, f, v, p)
                    z_hungarian_val.append(z_hungarian)
                    y_hungarian_val.append(y_hungarian)

                # Розрахуйте середні значення цільових функцій до списків
                avg_z_greedy = np.mean(z_greedy_val)
                avg_z_mak = np.mean(z_mak_val)
                avg_z_hungarian = np.mean(z_hungarian_val)

                avg_y_greedy = np.mean(y_greedy_val)
                avg_y_mak = np.mean(y_mak_val)
                avg_y_hungarian = np.mean(y_hungarian_val)

                # Додайте середні значення цільових функцій до списків
                avg_z_greedy_val.append(avg_z_greedy)
                avg_z_mak_val.append(avg_z_mak)
                avg_z_hungarian_val.append(avg_z_hungarian)

                avg_y_greedy_val.append(avg_y_greedy)
                avg_y_mak_val.append(avg_y_mak)
                avg_y_hungarian_val.append(avg_y_hungarian)

            # Створіть перший графік для середніх значень цільової функції z
            plt.plot(f_vars, avg_z_greedy_val, label='Greedy')
            plt.plot(f_vars, avg_z_mak_val, label='Mak')
            plt.plot(f_vars, avg_z_hungarian_val, label='Hungarian')
            plt.xlabel('f_var')
            plt.ylabel('Середнє значення цільової функції z')
            plt.title('Середнє значення цільової функції z для різних алгоритмів')
            plt.legend()
            plt.show()

            # Створіть другий графік для середніх значень другої цільової функції y
            plt.plot(f_vars, avg_y_greedy_val, label='Greedy')
            plt.plot(f_vars, avg_y_mak_val, label='Mak')
            plt.plot(f_vars, avg_y_hungarian_val, label='Hungarian')
            plt.xlabel('f_var')
            plt.ylabel('Середнє значення другої цільової функції y')
            plt.title('Середнє значення другої цільової функції y для різних алгоритмів')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра v_var (точність)
        if data_var_exp == 9:
            avg_z_greedy_val = []
            avg_z_mak_val = []
            avg_z_hungarian_val = []

            avg_y_greedy_val = []
            avg_y_mak_val = []
            avg_y_hungarian_val = []
            for v_var in v_vars:
                # Створіть списки для зберігання середніх значень цільових функцій
                z_greedy_val = []
                z_mak_val = []
                z_hungarian_val = []

                y_greedy_val = []
                y_mak_val = []
                y_hungarian_val = []

                for _ in range(20):
                    # Генерація задачі
                    n = 3
                    m = 3
                    f_mean = 0.5
                    f_var = 0.25
                    v_mean = 3
                    # v_var = 0.5
                    p_mean = 0.5
                    p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    f_rez, z_greedy, y_greedy = solve_greedy(m, n, f, v, p)
                    z_greedy_val.append(z_greedy)
                    y_greedy_val.append(y_greedy)

                    f_list = f.tolist()
                    f_rez, z_mak, y_mak = solve_mak(m, n, f_list, v, p)
                    z_mak_val.append(z_mak)
                    y_mak_val.append(y_mak)

                    f_rez, z_hungarian, y_hungarian = solve_hungarian(m, n, f, v, p)
                    z_hungarian_val.append(z_hungarian)
                    y_hungarian_val.append(y_hungarian)

                # Розрахуйте середні значення цільових функцій до списків
                avg_z_greedy = np.mean(z_greedy_val)
                avg_z_mak = np.mean(z_mak_val)
                avg_z_hungarian = np.mean(z_hungarian_val)

                avg_y_greedy = np.mean(y_greedy_val)
                avg_y_mak = np.mean(y_mak_val)
                avg_y_hungarian = np.mean(y_hungarian_val)

                # Додайте середні значення цільових функцій до списків
                avg_z_greedy_val.append(avg_z_greedy)
                avg_z_mak_val.append(avg_z_mak)
                avg_z_hungarian_val.append(avg_z_hungarian)

                avg_y_greedy_val.append(avg_y_greedy)
                avg_y_mak_val.append(avg_y_mak)
                avg_y_hungarian_val.append(avg_y_hungarian)

            # Створіть перший графік для середніх значень цільової функції z
            plt.plot(v_vars, avg_z_greedy_val, label='Greedy')
            plt.plot(v_vars, avg_z_mak_val, label='Mak')
            plt.plot(v_vars, avg_z_hungarian_val, label='Hungarian')
            plt.xlabel('v_var')
            plt.ylabel('Середнє значення цільової функції z')
            plt.title('Середнє значення цільової функції z для різних алгоритмів')
            plt.legend()
            plt.show()

            # Створіть другий графік для середніх значень другої цільової функції y
            plt.plot(v_vars, avg_y_greedy_val, label='Greedy')
            plt.plot(v_vars, avg_y_mak_val, label='Mak')
            plt.plot(v_vars, avg_y_hungarian_val, label='Hungarian')
            plt.xlabel('v_var')
            plt.ylabel('Середнє значення другої цільової функції y')
            plt.title('Середнє значення другої цільової функції y для різних алгоритмів')
            plt.legend()
            plt.show()

        # Проведення 20 експериментів для кожного значення параметра p_var (точність)
        if data_var_exp == 10:
            avg_z_greedy_val = []
            avg_z_mak_val = []
            avg_z_hungarian_val = []

            avg_y_greedy_val = []
            avg_y_mak_val = []
            avg_y_hungarian_val = []
            for p_var in p_vars:
                # Створіть списки для зберігання середніх значень цільових функцій
                z_greedy_val = []
                z_mak_val = []
                z_hungarian_val = []

                y_greedy_val = []
                y_mak_val = []
                y_hungarian_val = []

                for _ in range(20):
                    # Генерація задачі
                    n = 3
                    m = 3
                    f_mean = 0.5
                    f_var = 0.25
                    v_mean = 3
                    v_var = 0.5
                    p_mean = 0.5
                    # p_var = 0.25
                    f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

                    # Розв'язання задачі трьома алгоритмами
                    f_rez, z_greedy, y_greedy = solve_greedy(m, n, f, v, p)
                    z_greedy_val.append(z_greedy)
                    y_greedy_val.append(y_greedy)

                    f_list = f.tolist()
                    f_rez, z_mak, y_mak = solve_mak(m, n, f_list, v, p)
                    z_mak_val.append(z_mak)
                    y_mak_val.append(y_mak)

                    f_rez, z_hungarian, y_hungarian = solve_hungarian(m, n, f, v, p)
                    z_hungarian_val.append(z_hungarian)
                    y_hungarian_val.append(y_hungarian)

                # Розрахуйте середні значення цільових функцій до списків
                avg_z_greedy = np.mean(z_greedy_val)
                avg_z_mak = np.mean(z_mak_val)
                avg_z_hungarian = np.mean(z_hungarian_val)

                avg_y_greedy = np.mean(y_greedy_val)
                avg_y_mak = np.mean(y_mak_val)
                avg_y_hungarian = np.mean(y_hungarian_val)

                # Додайте середні значення цільових функцій до списків
                avg_z_greedy_val.append(avg_z_greedy)
                avg_z_mak_val.append(avg_z_mak)
                avg_z_hungarian_val.append(avg_z_hungarian)

                avg_y_greedy_val.append(avg_y_greedy)
                avg_y_mak_val.append(avg_y_mak)
                avg_y_hungarian_val.append(avg_y_hungarian)

            # Створіть перший графік для середніх значень цільової функції z
            plt.plot(p_vars, avg_z_greedy_val, label='Greedy')
            plt.plot(p_vars, avg_z_mak_val, label='Mak')
            plt.plot(p_vars, avg_z_hungarian_val, label='Hungarian')
            plt.xlabel('p_var')
            plt.ylabel('Середнє значення цільової функції z')
            plt.title('Середнє значення цільової функції z для різних алгоритмів')
            plt.legend()
            plt.show()

            # Створіть другий графік для середніх значень другої цільової функції y
            plt.plot(p_vars, avg_y_greedy_val, label='Greedy')
            plt.plot(p_vars, avg_y_mak_val, label='Mak')
            plt.plot(p_vars, avg_y_hungarian_val, label='Hungarian')
            plt.xlabel('p_var')
            plt.ylabel('Середнє значення другої цільової функції y')
            plt.title('Середнє значення другої цільової функції y для різних алгоритмів')
            plt.legend()
            plt.show()

# Порівняння алгоритмів
if data_var == 4:
    print('')
    print('-' * 50)
    print('\nПорівняння розв\'язку усіх алгоритмів.\n')
    print('Доступні опції:')
    options_com = ['згенерувати задачу',
                   'задати задачу',
                   'задача 1 (Павлова)',
                   'задача 2 (Гоголь)']
    for i in range(len(options_com)):
        print(f'{i + 1}. {options_com[i]}')
    data_var_com = int(input('mode:'))

    # print()
    # print('-' * 50)

    # Згенерувати
    if data_var_com == 1:
        print('\nАвтоматична генерація.')
        m = 3
        n = 3
        f_mean = 0.5
        f_var = 0.25
        v_mean = 3
        v_var = 0.5
        p_mean = 0.5
        p_var = 0.25

        f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

    # Задати вручну
    if data_var_com == 2:
        print('\nВизначення параметрів.\n')
        m = int(input('Кількість кандидатів (3-100)\nm = '))
        n = int(input('Кількість вакансій (3-14)\nn = '))

        f_mean = float(input('Мат. спод. еф. призн. f (0.1-1)\nf_mean = '))
        f_var = float(input('Дисп. еф. призн. f (0.1-1)\nf_var = '))

        v_mean = float(input('Мат. спод. квот v (1-20)\nv_mean = '))
        v_var = float(input('Дисп. квот v (0.1-1)\nv_var = '))

        p_mean = float(input('Мат. спод. пріорит. p (0.1-1)\np_mean = '))
        p_var = float(input('Дисп. пріорит. p (0.1-1)\np_var = '))

        f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)

    # Павлова
    if data_var_com == 3:
        num = 1
        name = 'Павлова'
        m = 5
        n = 3
        f = np.array(
            [[0.96, 0.91, 0.8, 0, 0.97],
             [0, 0.7, 0.51, 0.72, 0],
             [0.5, 0, 0.6, 0.6, 0.99]])
        v = [2, 1, 2]
        p = [0.5, 0.3, 0.2]

    # Гоголь
    if data_var_com == 4:
        num = 2
        name = 'Гоголь'
        m = 5
        n = 3
        f = np.array(
            [[0.63, 0.74, 0.1, 0, 0],
             [0, 0.7, 0.2, 0, 0.81],
             [0, 0, 0.9, 0.8, 0.39]])
        v = [3, 5, 2]
        p = [0.8, 0.5, 0.4]

    # Вхідна задача
    print()
    print('-' * 50)
    if data_var_com == 3 or data_var_com == 4:
        print(f'Задача {num} ({name})')
    else:
        print('Вхідна задача:')
    print('-' * 50)
    print(f'Кандидатів = {m}')
    print(f'Вакансій = {n}')
    print()
    print('v[i]\tp[i]\tf[i][j]')
    for i in range(len(v)):
        print(v[i], '\t\t', p[i], '\t', f[i])
    selected = [[] for _ in range(n)]
    print()

    # Жадібний
    print('-' * 50)
    print('Жадібний')
    print('-' * 50)
    f_rez, z, y = solve_greedy(m, n, f, v, p)
    # Результат
    print('Розподіл:')
    print(f'{f_rez}')
    print(f'z = {z}')
    print(f'y = {y}')
    print()

    # Мак
    print('-' * 50)
    print('Мак')
    print('-' * 50)
    f_list = f.tolist()
    f_rez, z, y = solve_mak(m, n, f_list, v, p)
    # Результат
    print('Розподіл:')
    print(f'{f_rez}')
    print(f'z = {z}')
    print(f'y = {y}')
    print()

    # Угорський
    print('-' * 50)
    print('Угорський')
    print('-' * 50)
    f_rez, z, y = solve_hungarian(m, n, f, v, p)
    # Результат
    print('Розподіл:')
    print(f'{f_rez}')
    print(f'z = {z}')
    print(f'y = {y}')

