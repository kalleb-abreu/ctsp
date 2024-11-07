import numpy as np
import time

class VNS_TSP:
    def __init__(self, instance, alpha=0.5, cycle=False, t_max=None):
        self.alpha = alpha
        if isinstance(instance, np.ndarray):
            self.distance_matrix = instance
        else:
            self.distance_matrix = instance.to_numpy()
        self.n = self.distance_matrix.shape[0]
        
        self.cycle = cycle

        if t_max is None:
            self.t_max = self.n
        else:
            self.t_max = t_max

        for i in range(self.n):
            self.distance_matrix[i][i] = np.inf
        self.RCL = self.generate_RCL()

        self.best_solution = []
        self.best_cost = np.inf

    def generate_RCL(self):
        mask = self.distance_matrix != np.inf
        min_dist = np.min(self.distance_matrix[mask])
        max_dist = np.max(self.distance_matrix[mask])
        threshold = min_dist + self.alpha * (max_dist - min_dist)

        return [
            np.max(self.distance_matrix[mask][i]) >= threshold for i in range(self.n)
        ]

    def remove_node(self, nodes, distance_matrix, solution):
        n1 = solution[-1]
        nodes = np.delete(nodes, np.where(nodes == n1))

        n2 = solution[-2]
        distance_matrix[n2, :] = np.inf
        distance_matrix[:, n2] = np.inf

        return nodes, distance_matrix

    def generate_solution(self):
        distance_matrix = self.distance_matrix.copy()
        nodes = np.array(np.arange(self.n))

        node = np.random.randint(self.n)
        nodes = np.delete(nodes, np.where(nodes == node))
        solution = [node]

        for _ in range(self.n - 1):
            if self.RCL[node]:
                node = nodes[np.random.randint(len(nodes))]
            else:
                node = np.argmin(distance_matrix[node])
            solution.append(node)
            nodes, distance_matrix = self.remove_node(nodes, distance_matrix, solution)

        return solution

    def objective_function(self, solution, distance_matrix=None):
        if distance_matrix is None:
            distance_matrix = self.distance_matrix
        cost = 0
        for i in range(self.n - 1):
            v_actual = solution[i]
            v_next = solution[i + 1]
            cost += distance_matrix[v_actual][v_next]
        v_actual = solution[-1]
        v_next = solution[0]
        if self.cycle:
            cost += distance_matrix[v_actual][v_next]
        return cost

    def swap(self, solution, i, j):
        new_solution = solution[:]
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def two_opt(self, solution, i, j):
        new_solution = solution[:i] + solution[i : j + 1][::-1] + solution[j + 1 :]
        return new_solution

    def shaking(self, solution, operator):
        n = len(solution)
        i, j = sorted(np.random.permutation(n)[:2])
        return operator(solution, i, j)

    def first_improvement(self, solution, operator):
        cost = self.objective_function(solution)
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                if j > i:
                    new_solution = operator(solution, i, j)
                    new_cost = self.objective_function(new_solution)
                    if new_cost < cost:
                        return new_solution, new_cost
        return solution, cost

    def vnd(self, solution, operators):
        cost = self.objective_function(solution, self.distance_matrix)

        l = 0
        l_max = len(operators)
        while l < l_max:
            new_solution, new_cost = self.first_improvement(solution, operators[l])
            if new_cost < cost:
                cost = new_cost
                solution = new_solution
                l = 0
            else:
                l += 1
            self.t_total = time.time() - self.t_init
            if self.t_total > self.t_max:
                break
        return solution, cost

    def fit(self):
        self.best_solution = self.generate_solution()
        self.best_cost = self.objective_function(self.best_solution)

        operators = [self.swap, self.two_opt]
        k_max = len(operators)

        self.t_init = time.time()
        while True:
            k = 0
            while k < k_max:
                new_solution = self.shaking(self.best_solution, operators[k])
                new_solution, new_cost = self.vnd(new_solution, operators)
                if new_cost < self.best_cost:
                    self.best_solution = new_solution
                    self.best_cost = new_cost
                    k = 0
                else:
                    k += 1
                self.t_total = time.time() - self.t_init
                if self.t_total > self.t_max:
                    break
            if self.t_total > self.t_max:
                break