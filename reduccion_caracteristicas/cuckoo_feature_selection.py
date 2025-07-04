import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import random

# Cargar datos
df = pd.read_csv("./dataset_caracteristicas_limpio.csv")
X = df.drop(columns=["archivo", "etiqueta"])
y = LabelEncoder().fit_transform(df["etiqueta"])

# Evaluación de precisión
def fitness(solution):
    if np.count_nonzero(solution) == 0:
        return 1.0  # Penalización por no seleccionar nada
    X_sel = X.iloc[:, solution == 1]
    acc = cross_val_score(SVC(kernel='linear'), X_sel, y, cv=5).mean()
    return 1 - acc  # Error a minimizar

# Inicialización
n_nests = 20
n_features = X.shape[1]
n_iterations = 50
pa = 0.25  # tasa de abandono
n_replace = int(pa * n_nests)

# Generar soluciones iniciales
population = np.random.randint(0, 2, (n_nests, n_features))
fitness_values = np.array([fitness(sol) for sol in population])

best_idx = np.argmin(fitness_values)
best_solution = population[best_idx].copy()
best_fitness = fitness_values[best_idx]

# Levy flight simplificado
def levy_flight(sol):
    new_sol = sol.copy()
    for i in range(len(sol)):
        if random.random() < 0.3:
            new_sol[i] = 1 - sol[i]  # cambiar bit
    return new_sol

# Algoritmo principal
for iter in range(n_iterations):
    for i in range(n_nests):
        new_sol = levy_flight(population[i])
        new_fit = fitness(new_sol)
        if new_fit < fitness_values[i]:
            population[i] = new_sol
            fitness_values[i] = new_fit
            if new_fit < best_fitness:
                best_solution = new_sol.copy()
                best_fitness = new_fit

    # Reemplazar algunos nidos
    worst_idx = fitness_values.argsort()[-n_replace:]
    for i in worst_idx:
        population[i] = np.random.randint(0, 2, n_features)
        fitness_values[i] = fitness(population[i])

# Resultados
selected_indices = np.where(best_solution == 1)[0]
selected_names = X.columns[selected_indices]


print("\nCaracterísticas seleccionadas:", selected_names.tolist())
print("Número de características:", len(selected_indices))
print("Precisión final con CCO:", 1 - best_fitness)

# Guardar dataset reducido con CCO
df_original = pd.read_csv('./dataset_caracteristicas_limpio.csv')
df_reducido = df_original[['archivo', 'etiqueta'] + list(selected_names)]
df_reducido.to_csv("dataset_cco_reducido.csv", index=False)
print("✅ Dataset reducido con CCO guardado como: dataset_cco_reducido.csv")
