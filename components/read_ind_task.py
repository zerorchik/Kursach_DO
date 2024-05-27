# -*- coding: utf-8 -*-

import numpy as np

# def read_individual_task(filename):
#     with open(filename, 'rb') as f_in:
#         m = np.load(f_in)
#         n = np.load(f_in)
#         f_matrix = np.load(f_in)
#         v_list = np.load(f_in)
#         p_list = np.load(f_in)
#     return m, n, f_matrix, v_list, p_list

def read_individual_task(filename):
    with open(filename, 'rb') as f_in:
        m = np.load(f_in)
        n = np.load(f_in)
        f_matrix = np.load(f_in, allow_pickle=True).tolist()
        v_list = np.load(f_in, allow_pickle=True).tolist()
        p_list = np.load(f_in, allow_pickle=True).tolist()
    return m, n, f_matrix, v_list, p_list

m, n, f_matrix, v_list, p_list = read_individual_task('task_params/p_var/task_m_3_n_3_f_mean_0.5_f_var_0.25_v_mean_3_v_var_0.5_p_mean_0.5_p_var_0.1.txt')

# print(f"m = {m}\n"
#       f"n = {n}\n"
#       f"f = [\n"
#       f"{f_matrix[0]}\n"
#       f"{f_matrix[1]}\n"
#       f"{f_matrix[2]}\n"
#       f"]\n"
#       f"v = {v_list}\n"
#       f"p = {p_list}")

print(f"m = {m}\n"
      f"n = {n}\n"
      f"f = [")
for i in range(m):
    row = f_matrix[i]
    row_str = " ".join(["{:.2f}".format(x) for x in row])
    print(f"  [{row_str}]")
print("]\n"
      f"v = {v_list}\n"
      f"p = {p_list}")