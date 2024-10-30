import numpy as np
from modules.vns_tsp import VNS_TSP

class CTSP:
    def __init__(self, instance, n_clusters):
        self.distance_matrix = instance.to_numpy()
        self.n_clusters = n_clusters        
        self.clusters = self.create_clusters_array()        
        self.clusters_order = np.arange(n_clusters)

    def create_clusters_array(self):
        clusters = [np.array([]).astype(int) for _ in range(self.n_clusters)]

        for i in range(self.distance_matrix.shape[0]):
            index = int(self.distance_matrix[i][0])
            clusters[index] = np.append(clusters[index], int(i))
        self.distance_matrix = self.distance_matrix[:, 1:]
        return clusters

    def get_short_distance_matrix(self, cluster_index):
        cluster = self.clusters[cluster_index]
        return self.distance_matrix[cluster, :][:, cluster]

    def solve_tsp(self):
        vns = VNS_TSP()

    def fit(self):
        print(self.clusters)
        for i in range(self.n_clusters):
            print(i)
        self.solve_tsp()
        
# istance_matrix = np.array[np.array] (0 - cluster, 1, 2, 3 ... custo do movimento)
# tsp_clusters = np.array[np.array, np.array, np.array]
# cluster_order = np.array[int, int, int, int]


# def solve_ctsp(distance_matrix):
#     return cost, tsp_cluster, cluster_order

# 	def solve_tsp(distance_matrix, cluster):
# 	   return cluster
	   
# 	def greedy_ctsp(distance_matrix, tsp_clusters):
#    	   return cluster_order
   	   
#    	def local_search_ctsp(distance_matrix, tsp_cluster, cluster_order):
#    	   return cluster_order
   	     
# def objective_function_tsp(cluster):
#     return cost
# ef objective_function_ctsp(tsp_cluster):
#    return cost