import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo
df = pd.read_csv("./extraccion_caracteristicas/resultados/audio_segmentado100m.csv")

# Seleccionar los primeros 10 fragmentos para graficar
df_subset = df[df['Segmento'] < 10]

# Graficar los 13 MFCCs para cada segmento
plt.figure(figsize=(14, 6))
for i in range(1, 14):
    plt.plot(df_subset['Segmento'], df_subset[f'MFCC_{i}'], label=f'MFCC {i}')

plt.title('Evolución de MFCCs en los primeros 10 segmentos de audio')
plt.xlabel('Segmento (10 segundos c/u)')
plt.ylabel('Valor promedio de MFCC')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
