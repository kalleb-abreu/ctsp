from modules.io import get_filenames, read_tsp_instance
from modules.ctsp import CTSP

ROOT_PATH = "./data/instances"
filenames = get_filenames(ROOT_PATH)

for i in range(len(filenames))[:1]:
    data = read_tsp_instance(filenames[i][0])

    ctsp = CTSP(data, n_clusters=filenames[i][1])
    ctsp.fit()

    # print(data)


#     instance = filename.split(ROOT_PATH)[-1][1:]
#     instances.append(instance)
#     print(f"{instance}")

#     tsp = GeneticAlgorithm(data, population_size=200, mutation_rate=0.02, num_generations=200)
#     tsp.evolve()

#     cost = tsp.best_cost
#     costs.append(cost)

#     it = tsp.total_iter
#     exec_time = tsp.total_time
#     print(f"   cost: {cost:7.4f}, iter: {it:4}, execution_time: {exec_time:12.2f}s\n")

#     solution = tsp.best_solution
#     solutions.append(solution)

# df = pd.DataFrame({"instance": instances, "best_OF": costs, "best_SOL": solutions})

# now_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# filename = f"./data/labs/lab7_ga/kalleb_abreu_TSP_ga_{now_str}.csv"

# df.to_csv(filename, index=False)