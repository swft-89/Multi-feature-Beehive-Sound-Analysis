# -*- coding: utf-8 -*-
"""
Clasificación supervisada con reducción de características (PSO y CCO)
Modelos: SVM y KNN
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

# Cargar etiquetas
y = pd.read_csv('dataset_clasificacion_pecoreo.csv')['Etiqueta']

# Cargar datasets reducidos
X_pso = pd.read_csv('X_pso_reducido.csv')
X_cco = pd.read_csv('X_cco_reducido.csv')

datasets = {
    'PSO': X_pso,
    'CCO': X_cco
}

modelos = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='linear', probability=True)
}

# Evaluar cada combinación
for nombre_ds, X in datasets.items():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    for nombre_mod, modelo in modelos.items():
        print(f"\n📘 Modelo: {nombre_mod} con {nombre_ds}")
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        print(classification_report(y_test, pred, digits=4))

        cm = confusion_matrix(y_test, pred, labels=modelo.classes_)
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        plt.title(f"Matriz de Confusión - {nombre_mod} con {nombre_ds}")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.show()

        scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring='f1_macro')
        print(f"F1 macro promedio ({nombre_mod} con {nombre_ds}): {scores.mean():.4f} ± {scores.std():.4f}")
