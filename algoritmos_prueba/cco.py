import numpy as np
import math
import matplotlib.pyplot as plt

class CCO:
    def __init__(self, func, dim, bounds, num_nests=25, max_iter=100, pa=0.25, step_size=1.0):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.num_nests = num_nests
        self.max_iter = max_iter
        self.pa = pa  # probability of discovery
        self.step_size = step_size
        
        # Initialize nests
        self.nests = np.random.uniform(bounds[0], bounds[1], (num_nests, dim))
        self.fitness = np.array([func(n) for n in self.nests])
        
        # Find best nest
        self.best_idx = np.argmin(self.fitness)
        self.best_nest = self.nests[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        
        # Store best values for plotting
        self.best_values = []
    
    def optimize(self):
        for _ in range(self.max_iter):
            # Generate new solutions via Lévy flights
            new_nests = self.nests.copy()
            for i in range(self.num_nests):
                step = self.step_size * self.levy_flight()
                new_nests[i] += step
                # Apply bounds
                new_nests[i] = np.clip(new_nests[i], self.bounds[0], self.bounds[1])
            
            # Evaluate new solutions
            new_fitness = np.array([self.func(n) for n in new_nests])
            
            # Replace worse nests with new better solutions
            improved_idx = new_fitness < self.fitness
            self.nests[improved_idx] = new_nests[improved_idx]
            self.fitness[improved_idx] = new_fitness[improved_idx]
            
            # Abandon some worse nests and build new ones
            abandon_mask = np.random.rand(self.num_nests) < self.pa
            num_abandon = np.sum(abandon_mask)
            if num_abandon > 0:
                self.nests[abandon_mask] = np.random.uniform(
                    self.bounds[0], self.bounds[1], (num_abandon, self.dim))
                self.fitness[abandon_mask] = np.array(
                    [self.func(n) for n in self.nests[abandon_mask]])
            
            # Find the best nest
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_nest = self.nests[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]
            
            self.best_values.append(self.best_fitness)
        
        return self.best_nest, self.best_fitness
    
    def levy_flight(self):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / \
                (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))**(1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / (np.abs(v)**(1 / beta))
        return step
    
    def plot_convergence(self):
        plt.plot(self.best_values)
        plt.title('Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.grid()
        plt.show()