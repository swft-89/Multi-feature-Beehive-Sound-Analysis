import numpy as np
import matplotlib.pyplot as plt

class PSO:
    def __init__(self, func, dim, bounds, num_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        
        # Initialize particles
        self.particles = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        
        # Initialize best positions
        self.pbest_pos = self.particles.copy()
        self.pbest_val = np.array([func(p) for p in self.particles])
        
        # Initialize global best
        self.gbest_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_val = self.pbest_val[self.gbest_idx]
        
        # Store best values for plotting
        self.best_values = []
    
    def optimize(self):
        for _ in range(self.max_iter):
            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            self.velocities = (self.w * self.velocities + 
                              self.c1 * r1 * (self.pbest_pos - self.particles) + 
                              self.c2 * r2 * (self.gbest_pos - self.particles))
            
            self.particles += self.velocities
            
            # Apply bounds
            self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])
            
            # Evaluate current positions
            current_val = np.array([self.func(p) for p in self.particles])
            
            # Update personal best
            improved_idx = current_val < self.pbest_val
            self.pbest_pos[improved_idx] = self.particles[improved_idx]
            self.pbest_val[improved_idx] = current_val[improved_idx]
            
            # Update global best
            current_best_idx = np.argmin(self.pbest_val)
            if self.pbest_val[current_best_idx] < self.gbest_val:
                self.gbest_pos = self.pbest_pos[current_best_idx].copy()
                self.gbest_val = self.pbest_val[current_best_idx]
            
            self.best_values.append(self.gbest_val)
        
        return self.gbest_pos, self.gbest_val
    
    def plot_convergence(self):
        plt.plot(self.best_values)
        plt.title('Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.grid()
        plt.show()

# Benchmark functions
def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)