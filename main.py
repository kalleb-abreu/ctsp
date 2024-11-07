from modules.io import (
    get_filenames,
    read_tsp_instance,
    get_results_file,
    save_instance_results
)

from modules.ctsp import CTSP
import os





ROOT_PATH = "./data/instances"
filenames = get_filenames(ROOT_PATH)
results_dir = "./data/results"
os.makedirs(results_dir, exist_ok=True)

for i in range(len(filenames))[:18]:
    data, df_coords = read_tsp_instance(filenames[i][0])

    ctsp = CTSP(data, n_clusters=filenames[i][1])
    ctsp.fit()

    instance = filenames[i][0].split(ROOT_PATH)[-1][1:]
    ctsp.print_results(instance)

    results_file = get_results_file(results_dir, i == 0)
    save_instance_results(results_file, instance, ctsp)

    output_path = os.path.join(results_dir, f"{instance}.png")
    ctsp.plot_solution(df_coords, output_path)
