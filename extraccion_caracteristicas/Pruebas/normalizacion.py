import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURACIÓN ---
ruta_csv_original = 'resultados/audio_segmentado100m.csv'  # CSV sin normalizar
ruta_csv_normalizado = 'resultados/audio_segmentado100m_normalizado.csv'  # CSV de salida

# --- CARGAR DATOS ---
df = pd.read_csv(ruta_csv_original)

# --- NORMALIZACIÓN ---
# 1. Separar la columna 'segmento' (no se normaliza)
columnas_caracteristicas = df.columns.drop('Segmento')  # Ej: ['mfcc_1', 'mfcc_2', ..., 'zcr']
caracteristicas = df[columnas_caracteristicas]

# 2. Aplicar normalización estándar (media=0, desviación=1)
scaler = StandardScaler()
caracteristicas_normalizadas = scaler.fit_transform(caracteristicas)

# 3. Reconstruir el DataFrame
df_normalizado = pd.DataFrame(caracteristicas_normalizadas, columns=columnas_caracteristicas)
df_normalizado.insert(0, 'Segmento', df['Segmento'])  # Agregar columna 'segmento'

# --- GUARDAR RESULTADO ---
os.makedirs("resultados", exist_ok=True)
df_normalizado.to_csv(ruta_csv_normalizado, index=False)

print("✅ ¡Normalización completada!")
print(f"Archivo original: {ruta_csv_original}")
print(f"Archivo normalizado guardado en: {ruta_csv_normalizado}")