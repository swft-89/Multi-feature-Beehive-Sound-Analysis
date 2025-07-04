import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler

# Cargar el dataset original
df = pd.read_csv("./dataset_caracteristicas_avanzadas.csv")

# Separar columnas que no se normalizan
columnas_no_numericas = ["archivo", "etiqueta"]
X = df.drop(columns=columnas_no_numericas)

# Corregir columnas que tienen strings como "[valor]"
for col in X.columns:
    if isinstance(X[col].iloc[0], str) and "[" in X[col].iloc[0]:
        try:
            X[col] = X[col].apply(lambda x: np.mean(ast.literal_eval(x)))
        except Exception as e:
            print(f"⚠️ Error en columna {col}: {e}")

# Convertir todo a float
X = X.astype(float)

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reconstruir el DataFrame final
df_clean = pd.DataFrame(X_scaled, columns=X.columns)
df_clean["archivo"] = df["archivo"]
df_clean["etiqueta"] = df["etiqueta"]

# Reordenar columnas
cols = ["archivo", "etiqueta"] + list(X.columns)
df_clean = df_clean[cols]

# Guardar el nuevo dataset limpio
df_clean.to_csv("dataset_caracteristicas_limpio.csv", index=False)
print("✅ Dataset limpio y normalizado guardado como: dataset_caracteristicas_limpio.csv")
