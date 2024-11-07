import numpy as np
import time
from modules.vns_tsp import VNS_TSP


class CTSP:
    def __init__(self, instance, n_clusters, time_max=None):
        self.distance_matrix = instance.to_numpy()
        self.n_clusters = n_clusters
        self.clusters = self.create_clusters_array()
        self.clusters_cost = [self.objective_function(
            self.clusters[index], self.distance_matrix) for index in range(self.n_clusters)]
        if time_max is None:
            self.max_time = self.n_clusters*self.n_clusters*self.distance_matrix.shape[0]
        else:
            self.max_time = time_max

    def create_clusters_array(self):
        clusters = [np.array([]).astype(int) for _ in range(self.n_clusters)]

        for i in range(self.distance_matrix.shape[0]):
            index = int(self.distance_matrix[i][0])
            clusters[index] = np.append(clusters[index], int(i))
        self.distance_matrix = self.distance_matrix[:, 1:]
        return clusters

    def objective_function(self, solution, distance_matrix=None, mode="ctsp"):
        if distance_matrix is None:
            distance_matrix = self.distance_matrix
        cost = 0
        n = len(solution)
        for i in range(n - 1):
            v_actual = solution[i]
            v_next = solution[i + 1]
            cost += distance_matrix[v_actual][v_next]
        v_actual = solution[-1]
        v_next = solution[0]
        if mode == "ctsp":
            cost += distance_matrix[v_actual][v_next]
        return cost

    def get_short_distance_matrix(self, cluster_index):
        cluster = self.clusters[cluster_index]
        return self.distance_matrix[cluster, :][:, cluster]

    def solve_tsp(self, distance_matrix):
        vns = VNS_TSP(distance_matrix)
        vns.fit()
        return np.array(vns.best_solution), vns.best_cost

    def greedy_ctsp(self):
        pool = np.arange(self.n_clusters)
        clusters_order = np.array([np.random.choice(pool)])

        pool = np.delete(pool, clusters_order[-1])
        while len(pool) > 0:
            min_cost = np.inf
            # check the cost in the distance matrix using the last value in the cluster_order
            for i in range(len(pool)):
                cost = self.distance_matrix[clusters_order[-1], pool[i]]
                if cost < min_cost:
                    min_cost = cost
                    min_cost_index = i
            # if the value is the minimum, add it to the cluster_order and remove it from the pool
            clusters_order = np.append(clusters_order, pool[min_cost_index])
            pool = np.delete(pool, min_cost_index)

        return clusters_order

    def get_ctsp_solution(self, clusters_order):
        solution = []
        for cluster in clusters_order:
            for node in self.clusters[cluster]:
                solution.append(int(node))
        return solution

    def build_initial_solution(self):
        for i in range(self.n_clusters):
            best_solution, best_cost = self.solve_tsp(
                self.get_short_distance_matrix(i))
            self.clusters[i] = self.clusters[i][best_solution]
            self.clusters_cost[i] = float(best_cost)

        self.clusters_order = self.greedy_ctsp()
        self.best_solution = self.get_ctsp_solution(self.clusters_order)
        self.best_cost = self.objective_function(
            self.best_solution, mode="ctsp")        
        self.initial_cost = self.best_cost

    def swap(self, solution, i, j):
        new_solution = solution[:]
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def two_opt(self, solution, i, j):
        new_solution = np.concatenate((
            solution[:i],
            solution[i:j+1][::-1],
            solution[j+1:]
        ))
        return new_solution

    def first_improvement(self, clusters_order, operator):
        solution = self.get_ctsp_solution(clusters_order)
        cost = self.objective_function(solution, mode='ctsp')
        for i in range(self.n_clusters - 1):
            for j in range(i + 1, self.n_clusters):
                if j > i:
                    new_clusters_order = operator(clusters_order, i, j)
                    new_solution = self.get_ctsp_solution(new_clusters_order)
                    new_cost = self.objective_function(
                        new_solution, mode='ctsp')
                    if new_cost < cost:
                        return new_clusters_order, new_solution, new_cost
        return clusters_order, solution, cost

    def shuffle_inner_clusters(self):
        n_shuffles = np.random.randint(1, self.n_clusters)
        for _ in range(n_shuffles):
            for i in range(self.n_clusters):
                self.clusters[i] = np.random.permutation(self.clusters[i])

    def local_search(self):
        max_not_improving = pow(self.distance_matrix.shape[0], 10)
        clusters_order = self.clusters_order
        self.not_improving = 0

        while True:
            while True: # se não melhorar, sai
                clusters_order, solution, cost = self.first_improvement(
                    clusters_order, self.two_opt)
                if cost < self.best_cost:
                    self.clusters_order = clusters_order
                    self.best_solution = solution
                    self.best_cost = cost
                else:
                    self.not_improving += 1
                    break
            condA = self.not_improving >= max_not_improving
            condB = time.time() - self.start_time - self.build_time > self.max_time
            
            if condA or condB:
                self.end_time = time.time()
                break
            
            self.shuffle_inner_clusters()

    def fit(self):
        self.start_time = time.time()
        self.build_initial_solution()
        self.build_time = time.time() - self.start_time

        self.local_search()
        self.total_time = self.end_time - self.start_time
        self.ls_time = self.total_time - self.build_time

    def print_results(self, instance, line_length=50):
        instance_display = f" {instance} "
        dashes = (line_length - len(instance_display)) // 2
        print(f"{'='*dashes}{instance_display}{'=' *
              (line_length-dashes-len(instance_display))}")
        print(f"Initial cost:{self.initial_cost:>{
              line_length-len('Initial cost:')},.4f}")
        print(f"Best cost found:{self.best_cost:>{
              line_length-len('Best cost found:')},.4f}")
        improvement = ((self.initial_cost - self.best_cost) / self.initial_cost) * 100
        print(f"Improvement (%):{improvement:>{
              line_length-len('Improvement (%):')},.2f}")
        print(f"Build time (s):{self.build_time:>{
              line_length-len('Build time (s):')},.2f}")
        print(f"Total execution time (s):{self.total_time:>{
              line_length-len('Total execution time (s):')},.2f}")
        print(f"Iterations without improvement:{self.not_improving:>{
              line_length-len('Iterations without improvement:')},}")
        print("-" * line_length, end="\n\n")


    def plot_solution(self, df, output_path):
        """
        Saves visualization of the clustered solution as a directed graph using node coordinates.
        Shows cluster regions and path between nodes based on cluster order.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing X,Y coordinates for each node
        output_path : str
            Path where the plot image should be saved
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull
        import numpy as np

        # Create directed graph
        G = nx.DiGraph()
        
        # Build solution following cluster order
        solution = []
        for i in range(len(self.clusters_order)):
            cluster_nodes = self.clusters[self.clusters_order[i]]
            solution.extend([int(node) for node in cluster_nodes])
            
            # Add edges between nodes within cluster
            for j in range(len(cluster_nodes)-1):
                G.add_edge(int(cluster_nodes[j]), int(cluster_nodes[j+1]))
            
            # If not last cluster, add edge to first node of next cluster
            if i < len(self.clusters_order)-1:
                next_cluster = self.clusters[self.clusters_order[i+1]]
                G.add_edge(int(cluster_nodes[-1]), int(next_cluster[0]))
        
        # Add edge from last node of last cluster to first node of first cluster
        first_cluster_first_node = int(self.clusters[self.clusters_order[0]][0])
        last_cluster_last_node = int(self.clusters[self.clusters_order[-1]][-1])
        G.add_edge(last_cluster_last_node, first_cluster_first_node)

        plt.figure(figsize=(12, 12))
        
        # Create position dictionary using X,Y coordinates
        pos = {node: (df.loc[node, 'X'], df.loc[node, 'Y']) 
               for node in G.nodes()}
        
        # Plot cluster regions in order
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.clusters)))
        
        # Create sorted cluster labels
        for i in range(len(self.clusters_order)):
            label = f'Cluster {self.clusters_order[i]}'
            cluster = self.clusters[self.clusters_order[i]]
            if len(cluster) > 2:  # Need at least 3 points for ConvexHull
                cluster_points = np.array([(df.loc[int(node), 'X'], df.loc[int(node), 'Y']) 
                                         for node in cluster])
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                
                # Create polygon with some transparency
                polygon = Polygon(hull_points, 
                                facecolor=colors[i],
                                alpha=0.2,
                                edgecolor=colors[i],
                                linewidth=2,
                                label=label)
                plt.gca().add_patch(polygon)
        
        # Draw nodes
        node_colors = ['red' if node == first_cluster_first_node else 'lightblue' 
                      for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, 
                             node_color=node_colors,
                             node_size=500)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos,
                             edge_color='gray',
                             width=2,
                             arrowsize=20,
                             arrowstyle='->',
                             connectionstyle='arc3,rad=0.2')
        
        # Add node labels
        node_labels = {node: f'{node}*' if node == first_cluster_first_node else str(node) 
                      for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        
        plt.title('Clustered Solution Path')
        plt.axis('equal')
        
        # Sort legend labels alphabetically
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0].split()[1])))
        plt.legend(handles, labels, title='Cluster Order')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()












