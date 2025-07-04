# -*- coding: utf-8 -*-
"""
Clasificación supervisada de segmentos de audio de colmena
Modelos: SVM y KNN
Datos: Todos los segmentos con etiquetas según distancia a fuente de alimento (100m, 300m, 500m)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Cargar dataset completo
df = pd.read_csv('dataset_clasificacion_pecoreo.csv')
X = df.drop(columns=['Segmento', 'Etiqueta'], errors='ignore')
y = df['Etiqueta']

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Modelos
modelos = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='linear', probability=True)
}

# Entrenamiento y evaluación
for nombre, modelo in modelos.items():
    print(f"\n📘 Modelo: {nombre}")
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    print(classification_report(y_test, pred, digits=4))

    cm = confusion_matrix(y_test, pred, labels=modelo.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring='f1_macro')
    print(f"F1 macro promedio (CV=5): {scores.mean():.4f} ± {scores.std():.4f}")
