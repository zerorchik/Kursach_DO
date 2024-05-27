# -*- coding: utf-8 -*-

import numpy as np

# def generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var):
#     # Генерація матриці ефективності призначення кандидатів на вакансії
#     f = np.random.normal(f_mean, f_var, (m, n))
#     f = np.maximum(0, np.minimum(1, f))
#
#     # Генерація квот вакансій
#     v = np.random.normal(v_mean, v_var, n)
#     v = np.maximum(1, np.round(v)).astype(int)
#
#     # Генерація пріоритетностей вакансій
#     p = np.random.normal(p_mean, p_var, n)
#     p = np.maximum(0, np.minimum(1, p))
#
#     return f, v, p

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

# def write_tasks_to_file(m, n, f, v, p, filename):
#     with open(filename, 'w') as f_out:
#         f_out.write(f"m = {m}\n")
#         f_out.write(f"n = {n}\n")
#         f_out.write("f = [\n")
#         for i in range(m):
#             f_out.write(" ".join(map(str, f[i])) + "\n")
#         f_out.write("]\n")
#         f_out.write("v = [\n")
#         f_out.write(" ".join(map(str, v)) + "\n")
#         f_out.write("]\n")
#         f_out.write("p = [\n")
#         f_out.write(" ".join(map(str, p)) + "\n")
#         f_out.write("]\n")

# def write_tasks_to_file(m, n, f, v, p, filename):
#     with open(filename, 'w') as f_out:
#         f_out.write(f"m = {m}\n")
#         f_out.write(f"n = {n}\n")
#         f_out.write("f = [\n")
#         for i in range(m):
#             f_out.write(" ".join(map(lambda x: "{:.2f}".format(x), f[i])) + "\n")
#         f_out.write("]\n")
#         f_out.write("v = [\n")
#         f_out.write(" ".join(map(lambda x: "{:.2f}".format(x), v)) + "\n")
#         f_out.write("]\n")
#         f_out.write("p = [\n")
#         f_out.write(" ".join(map(lambda x: "{:.2f}".format(x), p)) + "\n")
#         f_out.write("]\n")

def write_tasks_to_file(m, n, f, v, p, filename):
    with open(filename, 'wb') as f_out:
        np.save(f_out, m)
        np.save(f_out, n)
        np.save(f_out, f)
        np.save(f_out, v)
        np.save(f_out, p)

# Параметри задачі
# m_values = [3, 20, 50, 100]
# n_values = [3, 20, 50, 100]
# f_means = [0.1, 0.5, 1]
# f_vars = [0.01, 0.25, 0.5]
# v_means = [1, 8, 20]
# v_vars = [0.1, 0.5, 1]
# p_means = [0.2, 0.5, 0.8]
# p_vars = [0.1, 0.25, 0.5]

# Вакансії
# m = 3
# n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# f_mean = 0.5
# f_var = 0.25
# v_mean = 3
# v_var = 0.5
# p_mean = 0.5
# p_var = 0.25

# Кандидати
# m_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n = 3
# f_mean = 0.5
# f_var = 0.25
# v_mean = 3
# v_var = 0.5
# p_mean = 0.5
# p_var = 0.25

# # Математичне сподівання f
# m = 3
# n = 3
# f_means = [0.1, 0.5, 1]
# f_var = 0.25
# v_mean = 3
# v_var = 0.5
# p_mean = 0.5
# p_var = 0.25

# # Дисперсія f
# m = 3
# n = 3
# f_mean = 0.5
# f_vars = [0.01, 0.25, 0.5]
# v_mean = 3
# v_var = 0.5
# p_mean = 0.5
# p_var = 0.25

# # Математичне сподівання v
# m = 3
# n = 3
# f_mean = 0.5
# f_var = 0.25
# v_means = [1, 8, 20]
# v_var = 0.5
# p_mean = 0.5
# p_var = 0.25

# # Дисперсія v
# m = 3
# n = 3
# f_mean = 0.5
# f_var = 0.25
# v_mean = 3
# v_vars = [0.1, 0.5, 1]
# p_mean = 0.5
# p_var = 0.25

# # Математичне сподівання p
# m = 3
# n = 3
# f_mean = 0.5
# f_var = 0.25
# v_mean = 3
# v_var = 0.5
# p_means = [0.2, 0.5, 0.8]
# p_var = 0.25

# # Дисперсія p
# m = 3
# n = 3
# f_mean = 0.5
# f_var = 0.25
# v_mean = 3
# v_var = 0.5
# p_mean = 0.5
# p_vars = [0.1, 0.25, 0.5]

# Генерація
# for m in m_values:
#     for n in n_values:
#         for f_mean in f_means:
#             for f_var in f_vars:
#                 for v_mean in v_means:
#                     for v_var in v_vars:
#                         for p_mean in p_means:
#                             for p_var in p_vars:
#                                 f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#                                 filename = f"task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#                                 write_tasks_to_file(m, n, f, v, p, filename)
#                                 print(f"Згенеровано задачу {filename}")

# Вакансії
# for m in m_values:
#     for n in n_values:
#         f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#         filename = f"task_size/num_vacansies/task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#         write_tasks_to_file(m, n, f, v, p, filename)
#         print(f"Згенеровано задачу {filename}")

# Кандидати
# for m in m_values:
#     f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#     filename = f"task_size/num_candidates/task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#     write_tasks_to_file(m, n, f, v, p, filename)
#     print(f"Згенеровано задачу {filename}")

# # Математичне сподівання f
# for f_mean in f_means:
#     f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#     filename = f"task_params/f_mean/task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#     write_tasks_to_file(m, n, f, v, p, filename)
#     print(f"Згенеровано задачу {filename}")

# # Дисперсія f
# for f_var in f_vars:
#     f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#     filename = f"task_params/f_var/task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#     write_tasks_to_file(m, n, f, v, p, filename)
#     print(f"Згенеровано задачу {filename}")

# # Математичне сподівання v
# for v_mean in v_means:
#     f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#     filename = f"task_params/v_mean/task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#     write_tasks_to_file(m, n, f, v, p, filename)
#     print(f"Згенеровано задачу {filename}")

# # Дисперсія v
# for v_var in v_vars:
#     f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#     filename = f"task_params/v_var/task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#     write_tasks_to_file(m, n, f, v, p, filename)
#     print(f"Згенеровано задачу {filename}")

# # Математичне сподівання p
# for p_mean in p_means:
#     f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#     filename = f"task_params/p_mean/task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#     write_tasks_to_file(m, n, f, v, p, filename)
#     print(f"Згенеровано задачу {filename}")

# # Дисперсія p
# for p_var in p_vars:
#     f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
#     filename = f"task_params/p_var/task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
#     write_tasks_to_file(m, n, f, v, p, filename)
#     print(f"Згенеровано задачу {filename}")

# Параметри задачі
# m_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# f_means = [0.1, 0.5, 1]
# f_vars = [0.01, 0.25, 0.5]
# v_means = [1, 8, 20]
# v_vars = [0.1, 0.5, 1]
# p_means = [0.2, 0.5, 0.8]
# p_vars = [0.1, 0.25, 0.5]

# Універсал
m = 3
n = 3
f_mean = 0.5
f_var = 0.25
v_mean = 3
v_var = 0.5
p_mean = 0.5
p_var = 0.25

# Одинична
f, v, p = generate_individual_tasks(m, n, f_mean, f_var, v_mean, v_var, p_mean, p_var)
filename = f"task_m_{m}_n_{n}_f_mean_{f_mean}_f_var_{f_var}_v_mean_{v_mean}_v_var_{v_var}_p_mean_{p_mean}_p_var_{p_var}.txt"
write_tasks_to_file(m, n, f, v, p, filename)
print(f"Згенеровано задачу {filename}")