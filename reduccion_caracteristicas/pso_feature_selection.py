# pso_feature_selection.py
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pyswarms.discrete import BinaryPSO

# Leer dataset
df = pd.read_csv('./dataset_caracteristicas_limpio.csv')
X = df.drop(columns=['archivo', 'etiqueta'])
y = LabelEncoder().fit_transform(df['etiqueta'])

print(df.describe())


# Normalizar
X = StandardScaler().fit_transform(X)

# Función de evaluación (fitness)
def fitness_function(particles):
    scores = []
    for particle in particles:
        if np.count_nonzero(particle) == 0:
            scores.append(1.0)  # penalización
        else:
            X_subset = X[:, particle.astype(bool)]
            clf = SVC(kernel='linear')
            accuracy = cross_val_score(clf, X_subset, y, cv=5).mean()
            scores.append(1 - accuracy)  # minimizamos (1 - accuracy)
    return np.array(scores)


clf = SVC(kernel='linear')
score = cross_val_score(clf, X, y, cv=5).mean()
print("Precisión con todas las características:", score)

# PSO binario
options = {'c1': 2, 'c2': 2, 'w': 0.9, 'k': 5, 'p': 2}
n_particles = 20
dimensions = X.shape[1]
optimizer = BinaryPSO(n_particles=n_particles, dimensions=dimensions, options=options)


# Optimización
cost, pos = optimizer.optimize(fitness_function, iters=50)

print("Vector solución:", pos)
print("Cantidad de características activadas:", np.sum(pos))

# Resultados
selected_features = np.where(pos == 1)[0]
print("Características seleccionadas:", selected_features)
print("Número de características:", len(selected_features))

# Evaluación final
clf = SVC(kernel='linear')
X_selected = X[:, selected_features]
final_score = cross_val_score(clf, X_selected, y, cv=5).mean()
print("Precisión final con PSO:", final_score)

# Guardar dataset reducido con PSO
df_original = pd.read_csv('./dataset_caracteristicas_limpio.csv')
columnas_seleccionadas = df_original.drop(columns=['archivo', 'etiqueta']).columns[selected_features]
df_reducido = df_original[['archivo', 'etiqueta'] + list(columnas_seleccionadas)]
df_reducido.to_csv("dataset_pso_reducido.csv", index=False)
print("✅ Dataset reducido con PSO guardado como: dataset_pso_reducido.csv")
