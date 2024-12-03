import os

from modules.ctsp import CTSP
from modules.io import (
    get_filenames,
    read_tsp_instance,
    get_results_file,
    save_instance_results
)

ROOT_PATH = "./data/small_size"


def get_filenames(root_path):
    filenames = []
    for file in os.listdir(root_path):
        filenames.append((os.path.join(root_path, file), 3))
    return sorted(filenames)


filenames = get_filenames(ROOT_PATH)

cplex_results = {
    '5-eil51': (437, 12.31, 15.95),
    '10-eil51': (440, 74.38, 20.21),
    '15-eil51': (437, 2.04, None),
    '5-berlin52': (7991, 201.80, 22.48),
    '10-berlin52': (7096, 89.17, 10.87),
    '15-berlin52': (8049, 75.93, None),
    '5-st70': (695, 13790.11, 13.24),
    '10-st70': (691, 3831.00, 13.77),
    '15-st70': (692, 883.50, 16.02),
    '5-eil76': (559, 83.70, 9.28),
    '10-eil76': (561, 254.30, 18.78),
    '15-eil76': (565, 49.66, 10.27),
    '5-pr76': (108590, 99.29, 26.77),
    '10-pr76': (109538, 238.13, 13.23),
    '15-pr76': (110678, 261.94, 12.55),
    '10-rat99': (1238, 650.67, 9.98),
    '25-rat99': (1260, 351.15, None),
    '30-rat99': (1219, 2797.58, None),
    '25-kroA100': (21917, 3513.57, None),
    '50-kroA100': (21453, 947.55, None),
    '10-kroB100': (22440, 4991.44, 12.53),
    '50-kroB100': (22355, 2379.22, None),
    '25-eil101': (663, 709.45, None),
    '50-eil101': (644, 275.33, None),
    '25-lin105': (14838, 6224.55, None),
    '50-lin105': (14379, 1577.21, None),
    '75-lin105': (14521, 15886.77, None)





}

results_dir = "./output/results/tests"
graphs_dir = "./output/graphs/tests"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

flag = True
for k, v in cplex_results.items():
    filename = os.path.join(ROOT_PATH, f'{k}.csv')
    try:
        if v[2] is not None:
            data, df_coords = read_tsp_instance(filename)
            ctsp = CTSP(data, n_clusters=data['cluster'].max(
            )+1, tsp_max=0.1, time_max=min(v[1], 1), bks_cost=v[0])
            ctsp.fit()

            instance = filename.split(ROOT_PATH)[-1][1:]
            ctsp.print_results(instance, my_best_gap=v[2])

            results_file = get_results_file(results_dir, flag)
            save_instance_results(results_file, instance, ctsp)
            flag = False

            ctsp.plot_cost_over_time(instance.replace('.csv', ''), graphs_dir)

    except Exception as e:
        print(f"Error in instance {filename}: {e}")
