import numpy as np
from pso import PSO, sphere, rosenbrock
from cco import CCO

# Funciones benchmark adicionales
def rastrigin(x):
    """Rastrigin function - minimum at 0"""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    """Ackley function - minimum at 0"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    n = len(x)
    return -a * np.exp(-b * np.sqrt(sum_sq/n)) - np.exp(sum_cos/n) + a + np.exp(1)

def griewank(x):
    """Griewank function - minimum at 0"""
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return sum_part - prod_part + 1

def schwefel(x):
    """Schwefel function - minimum at ~420.9687 (for each dimension)"""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def zakharov(x):
    """Zakharov function - minimum at 0"""
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x)+1) * x)
    return sum1 + sum2**2 + sum2**4

def michalewicz(x, m=10):
    """Michalewicz function - minimum depends on dimension"""
    i = np.arange(1, len(x)+1)
    return -np.sum(np.sin(x) * np.sin((i * x**2)/np.pi)**(2*m))

def test_algorithms():
    # Configuración de pruebas
    test_cases = [
        {'name': 'Sphere (2D)', 'func': sphere, 'dim': 2, 'bounds': (-5, 5)},
        {'name': 'Sphere (10D)', 'func': sphere, 'dim': 10, 'bounds': (-5, 5)},
        {'name': 'Rosenbrock (2D)', 'func': rosenbrock, 'dim': 2, 'bounds': (-2, 2)},
        {'name': 'Rosenbrock (10D)', 'func': rosenbrock, 'dim': 10, 'bounds': (-2, 2)},
        {'name': 'Rastrigin (2D)', 'func': rastrigin, 'dim': 2, 'bounds': (-5.12, 5.12)},
        {'name': 'Rastrigin (10D)', 'func': rastrigin, 'dim': 10, 'bounds': (-5.12, 5.12)},
        {'name': 'Ackley (2D)', 'func': ackley, 'dim': 2, 'bounds': (-32, 32)},
        {'name': 'Ackley (10D)', 'func': ackley, 'dim': 10, 'bounds': (-32, 32)},
        {'name': 'Griewank (2D)', 'func': griewank, 'dim': 2, 'bounds': (-600, 600)},
        {'name': 'Griewank (10D)', 'func': griewank, 'dim': 10, 'bounds': (-600, 600)},
        {'name': 'Schwefel (2D)', 'func': schwefel, 'dim': 2, 'bounds': (-500, 500)},
        {'name': 'Schwefel (10D)', 'func': schwefel, 'dim': 10, 'bounds': (-500, 500)},
        {'name': 'Zakharov (2D)', 'func': zakharov, 'dim': 2, 'bounds': (-5, 10)},
        {'name': 'Zakharov (10D)', 'func': zakharov, 'dim': 10, 'bounds': (-5, 10)},
        {'name': 'Michalewicz (2D)', 'func': michalewicz, 'dim': 2, 'bounds': (0, np.pi)},
        {'name': 'Michalewicz (10D)', 'func': michalewicz, 'dim': 10, 'bounds': (0, np.pi)}
    ]
    
    for case in test_cases:
        print(f"\nTesting {case['name']}")
        
        # PSO
        print("\nRunning PSO...")
        pso = PSO(case['func'], case['dim'], case['bounds'])
        best_pos, best_val = pso.optimize()
        print(f"PSO - Best Position: {best_pos}")
        print(f"PSO - Best Value: {best_val:.6f}")
        pso.plot_convergence()
        
        # CCO
        print("\nRunning CCO...")
        cco = CCO(case['func'], case['dim'], case['bounds'])
        best_pos, best_val = cco.optimize()
        print(f"CCO - Best Position: {best_pos}")
        print(f"CCO - Best Value: {best_val:.6f}")
        cco.plot_convergence()

if __name__ == "__main__":
    test_algorithms()