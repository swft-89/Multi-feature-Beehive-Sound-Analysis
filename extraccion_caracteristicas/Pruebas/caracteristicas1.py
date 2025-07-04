import librosa
import numpy as np
import pandas as pd
import os

# --- CONFIGURACIÓN ---
ruta_audio = 'audios_colmena/100m.wav'  
duracion_segmento = 10  # Duración en segundos por fragmento
n_mfcc = 13  # Número de coeficientes MFCC a extraer

# --- CARGAR AUDIO ---
y, sr = librosa.load(ruta_audio, sr=None)
longitud_segmento = duracion_segmento * sr

print(f"Duración total: {len(y)/sr:.2f} segundos")
print(f"Frecuencia de muestreo: {sr} Hz")

# --- DIVIDIR AUDIO EN FRAGMENTOS ---
fragmentos = [y[i:i+longitud_segmento] for i in range(0, len(y), longitud_segmento)]

# --- EXTRAER CARACTERÍSTICAS ---
caracteristicas_fragmentos = []

for idx, fragmento in enumerate(fragmentos):
    if len(fragmento) < longitud_segmento:
        continue  # Saltar fragmentos incompletos

    # MFCC
    mfcc = librosa.feature.mfcc(y=fragmento, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(fragmento)[0]
    zcr_mean = np.mean(zcr)
    
    # Chroma (12 valores, uno por nota musical)
    chroma = librosa.feature.chroma_stft(y=fragmento, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral Centroid (1 valor)
    centroid = librosa.feature.spectral_centroid(y=fragmento, sr=sr)
    centroid_mean = np.mean(centroid)

    # Guardar características
    vector = np.concatenate((
        [idx], 
        mfcc_mean, 
        [zcr_mean], 
        chroma_mean, 
        [centroid_mean]
    ))
    caracteristicas_fragmentos.append(vector)

# --- GUARDAR EN CSV ---
# Crear nombres de columnas
columnas = (['Segmento'] + 
           [f'MFCC_{i+1}' for i in range(n_mfcc)] + 
           ['ZCR'] + 
           [f'Chroma_{i+1}' for i in range(12)] + 
           ['Spectral_Centroid'])
df = pd.DataFrame(caracteristicas_fragmentos, columns=columnas)

# Crear carpeta si no existe
os.makedirs("resultados", exist_ok=True)

# Guardar
df.to_csv('resultados/audio_segmentado100m.csv', index=False)
print("✅ ¡Características guardadas exitosamente en resultados/audio_segmentado.csv!")
