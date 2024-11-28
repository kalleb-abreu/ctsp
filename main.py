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
    '5-eil51': (437, 12.31),
    '10-eil51': (440, 74.38),
    '15-eil51': (437, 2.04),
    '5-berlin52': (7991, 201.80),
    '10-berlin52': (7096, 89.17),
    '15-berlin52': (8049, 75.93),
    '5-st70': (695, 13790.11),
    '10-st70': (691, 3831.00),
    '15-st70': (692, 883.50),
    '5-eil76': (559, 83.70),
    '10-eil76': (561, 254.30),
    '15-eil76': (565, 49.66),
    '5-pr76': (108590, 99.29),
    '10-pr76': (109538, 238.13),
    '15-pr76': (110678, 261.94),
    '10-rat99': (1238, 650.67),
    '25-rat99': (1260, 351.15),
    '30-rat99': (1219, 2797.58),
    '25-kroA100': (21917, 3513.57),
    '50-kroA100': (21453, 947.55),
    '10-kroB100': (22440, 4991.44),
    '50-kroB100': (22355, 2379.22),
    '25-eil101': (663, 709.45),
    '50-eil101': (644, 275.33),
    '25-lin105': (14838, 6224.55),
    '50-lin105': (14379, 1577.21),
    '75-lin105': (14521, 15886.77)
}

results_dir = "./output/results"
graphs_dir = "./output/graphs"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

flag = True
for k, v in cplex_results.items():
    filename = os.path.join(ROOT_PATH, f'{k}.csv')
    try:
        data, df_coords = read_tsp_instance(filename)
        ctsp = CTSP(data, n_clusters=data['cluster'].max(
        )+1, tsp_max=0.1, time_max=min(v[1], 200), bks_cost=v[0])
        ctsp.fit()

        instance = filename.split(ROOT_PATH)[-1][1:]
        ctsp.print_results(instance)

        results_file = get_results_file(results_dir, flag)
        save_instance_results(results_file, instance, ctsp)
        flag = False

        ctsp.plot_cost_over_time(instance.replace('.csv', ''), graphs_dir)

    except Exception as e:
        print(f"Error in instance {filename}: {e}")
