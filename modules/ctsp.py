import matplotlib.pyplot as plt
import numpy as np
import time
import os

from modules.vns_tsp import VNS_TSP

# TODO:
# - Refactor to use classes for each component (SolutionBuilder, LocalSearch, Results, Configuration)
# - Move utility functions to separate module (utils.py)
# - Consider using a Configuration class to handle all parameters   

class CTSP:
    def __init__(self, instance, n_clusters, bks_cost=None, tsp_max=None, time_max=None, gap=None):
        
        # Distance Matrix
        self.distance_matrix = instance.to_numpy()
        
        # Number of Clusters
        self.n_clusters = n_clusters
        
        # Clusters
        self.clusters = self.create_clusters_array()
        
        # Order of Clusters
        self.clusters_order = np.arange(self.n_clusters)

        self.clusters_cost = [self.objective_function(
            self.clusters[index], self.distance_matrix) for index in range(self.n_clusters)]
        
        self.best_cost = np.inf
        self.best_solution = None
                
        if time_max is None:
            self.max_time = int(4.5 * self.distance_matrix.shape[0])
        else:
            self.max_time = time_max
        if tsp_max is None:
            self.tsp_max = int(self.distance_matrix.shape[0]/10)
        else:
            self.tsp_max = tsp_max
        self.gap = gap
        self.bks_cost = bks_cost
        self.local_search_time = 0
        self.time_history = []
        self.best_cost_history = []
        self.last_checkpoint = 0

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
        vns = VNS_TSP(distance_matrix, t_max=self.tsp_max)
        vns.fit()
        return np.array(vns.best_solution), vns.best_cost

    def greedy_ctsp(self, first=None):
        pool = np.arange(self.n_clusters)
        if first is None:
            first = np.random.choice(pool)
        pool = np.delete(pool, first)
        clusters_order = np.array([first])
        
        while len(pool) > 0:
            min_cost = np.inf
            for i in range(len(pool)):
                last_node = self.clusters[clusters_order[-1]][-1]
                next_node = self.clusters[pool[i]][0]
                cost = self.distance_matrix[last_node, next_node]
                if cost < min_cost:
                    min_cost = cost
                    min_cost_index = i
            clusters_order = np.append(clusters_order, pool[min_cost_index])
            pool = np.delete(pool, min_cost_index)

        return clusters_order

    def get_ctsp_solution(self, clusters_order):
        solution = []
        for cluster in clusters_order:
            for node in self.clusters[cluster]:
                solution.append(int(node))
        return solution
    
    def fixed_starting_node(self, order):
        zero_index = np.where(order == 0)[0][0]
        if zero_index == 0:
            return order
        return np.concatenate((order[zero_index:], order[:zero_index]))

    def check_cyclic_solution(self, orders):
        fixed_orders = []
        for order in orders:
            fixed_orders.append(self.fixed_starting_node(order))        
        index = []
        for i in range(len(fixed_orders)-1, -1, -1):
            for j in range(i):
                flag = True
                for k in range(self.n_clusters):
                    if fixed_orders[i][k] != fixed_orders[j][k]:
                        flag = False
                        break
                if flag:
                    if i not in index:
                        index.append(i)
        for i in index:
            orders.pop(i)
        return orders

    def build_solution(self):
        for i in range(self.n_clusters):
            best_solution, best_cost = self.solve_tsp(
                self.get_short_distance_matrix(i))
            self.clusters[i] = self.clusters[i][best_solution]
            self.clusters_cost[i] = float(best_cost)

        clusters_orders = []
        for i in range(self.n_clusters):
            clusters_orders.append(self.greedy_ctsp(first=i))
        
        self.check_cyclic_solution(clusters_orders)
        
        self.time_history.append(0)
        self.best_cost_history.append(self.best_cost)

        return self.clusters.copy(), self.clusters_cost.copy(), clusters_orders

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
                    if new_cost < self.best_cost:
                        return new_clusters_order, new_solution, new_cost
        return clusters_order, solution, cost
    
    def get_mask_combinations(self, n_max=20):
        n = min(self.n_clusters, n_max) if n_max is not None else self.n_clusters
        combinations = []
        for x in range(2**n):
            mask = [int(b) for b in f'{x:0{n}b}']
            if len(mask) < self.n_clusters:
                mask.extend([0] * (self.n_clusters - len(mask)))
            combinations.append(mask)
        return np.array(combinations)
    
    def reverse_clusters_by_mask(self, mask):
        """Reverse clusters according to mask values (1=reverse, 0=keep)"""
        for i, should_reverse in enumerate(mask):
            if should_reverse:
                self.clusters[i] = self.clusters[i][::-1]

    def local_search(self):
        clusters_order = self.clusters_order.copy()
        masks = self.get_mask_combinations()            
        while True:     
            for i in range(len(masks)):
                self.reverse_clusters_by_mask(masks[i])
                cost = self.objective_function(self.get_ctsp_solution(clusters_order), mode='ctsp')
                while True: # se não melhorar, sai
                    current_time = time.time() - self.ls_start_time
                    if current_time - self.last_checkpoint >= 1:
                        self.time_history.append(current_time)
                        self.best_cost_history.append(self.best_cost)
                        self.last_checkpoint = current_time
                        # print(f"Time: {current_time:.2f}s, Cost: {self.best_cost:.4f}")
                    clusters_order, solution, cost = self.first_improvement(
                        clusters_order, self.two_opt)
                    if cost < self.best_cost:
                        self.clusters_order = clusters_order
                        self.best_solution = solution
                        self.best_cost = cost
                        i = 0
                    else:
                        self.reverse_clusters_by_mask(masks[i])
                        break
                                            # Record time and cost every 10 seconds
 
            local_search_time = time.time() - self.ls_start_time
            # condA = self.gap is not None and self.bks_cost is not None and cost <= self.bks_cost * (1 + self.gap)
            # condB = self.max_time is not None and local_search_time + self.build_time > self.max_time
            
            # if condA or condB:
            self.end_time = time.time()
            self.local_search_time = self.end_time - self.ls_start_time
            
            # Record final time and cost
            if self.time_history[-1] != local_search_time:
                self.time_history.append(local_search_time)
                self.best_cost_history.append(self.best_cost)
            break

    def fit(self):

        self.start_time = time.time()
        clusters, clusters_cost, clusters_orders = self.build_solution()
        self.build_time = time.time() - self.start_time

        self.ls_start_time = time.time()
        for i in range(len(clusters_orders)):
            self.clusters_order = clusters_orders[i]
            self.clusters = clusters.copy()
            self.clusters_cost = clusters_cost.copy()

            cost = self.objective_function(self.get_ctsp_solution(self.clusters_order), mode="ctsp")
            if cost < self.best_cost:
                self.best_solution = self.get_ctsp_solution(self.clusters_order)
                self.best_cost = cost
            if i == 0:
                self.initial_cost = self.best_cost

            self.local_search()
        self.total_time = self.end_time - self.start_time

    def print_results(self, instance, my_best_gap=None, line_length=50):
        instance_display = f" {instance} "
        dashes = (line_length - len(instance_display)) // 2
        print(f"{'='*dashes}{instance_display}{'=' * (line_length-dashes-len(instance_display))}")
        if self.bks_cost is not None:
            print(f"BKS cost:{self.bks_cost:>{line_length-len('BKS cost:')},.4f}")
        print(f"Initial cost:{self.initial_cost:>{line_length-len('Initial cost:')},.4f}")
        print(f"Best cost found:{self.best_cost:>{line_length-len('Best cost found:')},.4f}")
        improvement = ((self.initial_cost - self.best_cost) / self.initial_cost) * 100
        print(f"Improvement from initial (%):{improvement:>{line_length-len('Improvement from initial (%):')},.2f}")
        if self.bks_cost is not None:
            gap = ((self.best_cost - self.bks_cost) / self.bks_cost) * 100
            gap_improvement = my_best_gap - gap
            print(f"Gap to BKS (%):{gap:>{line_length-len('Gap to BKS (%):')},.2f}")
            print(f"Gap improvement (%):{gap_improvement:>{line_length-len('Gap improvement (%):')},.2f}")
        print(f"Build time (s):{self.build_time:>{line_length-len('Build time (s):')},.2f}")
        print(f"Local search time (s):{self.local_search_time:>{line_length-len('Local search time (s):')},.2f}")
        print(f"Total execution time (s):{self.total_time:>{line_length-len('Total execution time (s):')},.2f}")
        print("-" * line_length, end="\n\n")

    def plot_shuffle_probability(self, exponents=[2, 3, 4, 5]):
        times = np.linspace(0, self.max_time, 100)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        for exp in exponents:
            probs = [self.get_shuffle_probability(t, exponent=exp) for t in times]
            plt.plot(times, probs, label=f'exponent={exp}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Shuffle Probability')
        plt.title('Shuffle Probability vs Time')
        plt.grid(True)
        plt.legend()
        plt.savefig('shuffle_prob.png')
        plt.close()

    def plot_cost_over_time(self, instance, output_path):
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot cost in upper subplot
        ax1.plot(self.time_history, self.best_cost_history, 'b-')
        ax1.set_ylabel('Best Cost Found')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True)
        ax1.set_title(f'Cost Evolution Over Time - {instance}')
        
        # Plot gap in lower subplot if BKS exists
        if self.bks_cost is not None:
            gaps = [(cost - self.bks_cost) / self.bks_cost * 100 
                   for cost in self.best_cost_history]
            ax2.plot(self.time_history, gaps, 'r-')
            ax2.set_ylabel('Gap to BKS (%)')
        else:
            ax2.text(0.5, 0.5, 'No BKS available', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes)
            
        ax2.grid(True)
        ax2.set_xlabel('Time (s)')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'cost_over_time_{instance}.png'))
        plt.close()