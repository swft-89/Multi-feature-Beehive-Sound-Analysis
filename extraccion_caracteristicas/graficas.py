from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

# Cargar dataset
df = pd.read_csv("./extraccion_caracteristicas/resultados/dataset_caracteristicas_limpio.csv")
X = df.drop(columns=["archivo", "etiqueta"])
y = LabelEncoder().fit_transform(df["etiqueta"])

# Normalizar
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("PCA - Nuevas características")
plt.colorbar()
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("t-SNE - Nuevas características")
plt.colorbar()
plt.show()
