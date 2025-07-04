import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Cargar el dataset
df = pd.read_csv('./extraccion_caracteristicas/resultados/Pruebas/dataset_clasificacion_pecoreo.csv')

# Verificar etiquetas
print("Distribución de clases:", df['Etiqueta'].value_counts())

# Separar características y etiquetas
X = df.drop(columns=['Segmento', 'Etiqueta'])
y = LabelEncoder().fit_transform(df['Etiqueta'])

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("Visualización PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Etiqueta codificada")
plt.grid(True)
plt.tight_layout()
plt.show()

# t-SNE (2D)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("Visualización t-SNE (2D)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar(label="Etiqueta codificada")
plt.grid(True)
plt.tight_layout()
plt.show()

# Opcional: Silhouette Score (qué tan bien están separados los grupos)
sil_score = silhouette_score(X_scaled, y)
print(f"Silhouette Score (mayor es mejor separación): {sil_score:.4f}")

# Gráfico adicional: boxplot de algunas características
df_viz = df.copy()
df_viz['Etiqueta'] = df_viz['Etiqueta'].astype(str)

plt.figure(figsize=(10,5))
sns.boxplot(data=df_viz, x='Etiqueta', y='MFCC_1')
plt.title('Distribución de MFCC_1 por clase')
plt.grid(True)
plt.tight_layout()
plt.show()

