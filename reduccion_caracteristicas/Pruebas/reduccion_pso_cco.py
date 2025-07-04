# --- IMPORTACIONES ---
import pandas as pd
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

# --- CARGAR DATOS ---
print("📥 Cargando archivo: audio_segmentado500m_normalizado.csv")
df = pd.read_csv('./extraccion_caracteristicas/resultados/audio_segmentado500m_normalizado.csv')
X = df.drop(columns=['Segmento'])
dim = X.shape[1]  
print(f"🔎 Total de características: {dim}")

# --- FUNCIÓN FITNESS NO SUPERVISADA ---
def fitness_unsupervised(position, X_train):
    selected = [i for i, bit in enumerate(position) if bit == 1]
    if not selected:
        return 0.0  # Evitar selección vacía
    
    X_subset = X_train.iloc[:, selected]
    
    # 1. Maximizar varianza (características informativas)
    var_score = np.mean(X_subset.var())  # Promedio de varianzas
    
    # 2. Minimizar redundancia (correlación entre características)
    corr_matrix = np.corrcoef(X_subset, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)  # Ignorar autocorrelación
    redundancy_penalty = np.mean(np.abs(corr_matrix))  # Promedio de correlaciones
    
    # Fitness = Varianza - Redundancia (pesos ajustables)
    fitness = var_score - (0.5 * redundancy_penalty)  
    
    return fitness

# --- ALGORITMO PSO (No Supervisado) ---
def pso(num_particles=10, num_iterations=20):
    particles = [np.random.randint(0, 2, dim).tolist() for _ in range(num_particles)]
    velocities = [np.zeros(dim).tolist() for _ in range(num_particles)]
    pbest = particles.copy()
    pbest_scores = [fitness_unsupervised(p, X) for p in particles]  # Usamos X completo (no supervisado)
    gbest = pbest[np.argmax(pbest_scores)]
    gbest_score = max(pbest_scores)

    for _ in range(num_iterations):
        for i in range(num_particles):
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (0.5 * velocities[i][d] +
                                    1.5 * r1 * (pbest[i][d] - particles[i][d]) +
                                    1.5 * r2 * (gbest[d] - particles[i][d]))
                sigmoid = 1 / (1 + np.exp(-velocities[i][d]))
                particles[i][d] = 1 if random.random() < sigmoid else 0
            score = fitness_unsupervised(particles[i], X)
            if score > pbest_scores[i]:
                pbest[i] = particles[i]
                pbest_scores[i] = score
        gbest = pbest[np.argmax(pbest_scores)]
        gbest_score = max(pbest_scores)

    return gbest, gbest_score

# --- ALGORITMO CCO (No Supervisado) ---
def cco(num_agents=10, num_iterations=20):
    population = [np.random.randint(0, 2, dim).tolist() for _ in range(num_agents)]
    fitness = [fitness_unsupervised(p, X) for p in population]
    best_idx = np.argmax(fitness)
    best_sol = population[best_idx]
    best_score = fitness[best_idx]

    for _ in range(num_iterations):
        for i in range(num_agents):
            new_sol = [(bit if random.random() > 0.3 else 1 - bit) for bit in population[i]]
            score = fitness_unsupervised(new_sol, X)
            if score > fitness[i]:
                population[i] = new_sol
                fitness[i] = score

        best_idx = np.argmax(fitness)
        best_sol = population[best_idx]
        best_score = fitness[best_idx]

    return best_sol, best_score

# --- EJECUTAR PSO ---
print("\n🚀 Ejecutando PSO (no supervisado)...")
start_pso = time.time()
pso_solution, pso_score = pso()
end_pso = time.time()
pso_selected = [i for i, v in enumerate(pso_solution) if v == 1]
print(f"✅ PSO Fitness Score: {pso_score:.4f}")
print(f"📉 Características seleccionadas por PSO ({len(pso_selected)}): {pso_selected}")
print(f"⏱ Tiempo de cómputo PSO: {end_pso - start_pso:.2f} segundos")

# --- EJECUTAR CCO ---
print("\n🦍 Ejecutando CCO (no supervisado)...")
start_cco = time.time()
cco_solution, cco_score = cco()
end_cco = time.time()
cco_selected = [i for i, v in enumerate(cco_solution) if v == 1]
print(f"✅ CCO Fitness Score: {cco_score:.4f}")
print(f"📉 Características seleccionadas por CCO ({len(cco_selected)}): {cco_selected}")
print(f"⏱ Tiempo de cómputo CCO: {end_cco - start_cco:.2f} segundos")

# --- GUARDAR RESULTADOS ---
os.makedirs("resultados", exist_ok=True)

X.iloc[:, pso_selected].to_csv("./reduccion_caracteristicas/resultados/500m/pso_reducido_500m.csv", index=False)
X.iloc[:, cco_selected].to_csv("./reduccion_caracteristicas/resultados/500m/cco_reducido_500m.csv", index=False)
pd.Series(pso_selected).to_csv("./reduccion_caracteristicas/resultados/500m/indices_pso500m.csv", index=False, header=['columna'])
pd.Series(cco_selected).to_csv("./reduccion_caracteristicas/resultados/500m/indices_cco500m.csv", index=False, header=['columna'])

print("\n📁 Reducción de características guardada en la carpeta 'resultados/'")

# --- GRÁFICA COMPARATIVA ---
plt.figure(figsize=(6, 4))
plt.bar(['PSO', 'CCO'], [pso_score, cco_score], color=['skyblue', 'salmon'])
plt.title('Comparación de Fitness (No Supervisado)')
plt.ylabel('Fitness Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()