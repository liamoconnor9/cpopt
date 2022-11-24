import pickle
import numpy as np
import matplotlib.pyplot as plt

# run_names = ['triangular_cone_1', 'parabolic_cone_1', 'elliptic_cone_1']

run_names = ['triangular_cone']
metrics_lst = []

for run_name in run_names:
    with open(run_name + '/metrics.pick', 'rb') as file:
        metrics_dictionary = pickle.load(file)
        metrics_lst.append(metrics_dictionary)

for metrics_dictionary in metrics_lst:
    drag_lst = metrics_dictionary['drag_lst']
    times_lst = metrics_dictionary['time_lst']
    plt.plot(times_lst, drag_lst)
    plt.yscale('log')

plt.show()