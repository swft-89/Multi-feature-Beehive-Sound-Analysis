import librosa
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Ruta raíz que contiene las subcarpetas
ROOT_DIR = "./audios_segmentados"

print("🚀 Iniciando extracción de características...")

# Lista para almacenar todos los datos
features_list = []

# Recorrer subcarpetas (una por clase)
for folder in sorted(os.listdir(ROOT_DIR)):
    folder_path = os.path.join(ROOT_DIR, folder)

    if os.path.isdir(folder_path):
        etiqueta = folder.split("_")[-1]  # Obtiene 100m, 300m, 500m

        for archivo in sorted(os.listdir(folder_path)):
            if archivo.endswith(".wav"):
                file_path = os.path.join(folder_path, archivo)
                print("Procesando:", file_path)
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    
                    if len(y) < 1024:
                        print(f"⚠️ Archivo omitido por ser muy corto: {file_path}")
                        continue
 
                    feature_dict = {"archivo": archivo, "etiqueta": etiqueta}

                    # MFCCs
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    for i in range(mfcc.shape[0]):
                        feature_dict[f"MFCC_{i+1}"] = np.mean(mfcc[i])

                    # Zero Crossing Rate
                    zcr = librosa.feature.zero_crossing_rate(y)
                    feature_dict["ZCR"] = np.mean(zcr)

                    # RMS
                    rms = librosa.feature.rms(y=y)
                    feature_dict["RMS"] = np.mean(rms)

                    # Spectral features
                    feature_dict["Centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    feature_dict["Bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                    feature_dict["Rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                    feature_dict["Flatness"] = np.mean(librosa.feature.spectral_flatness(y=y))

                    # Chroma STFT
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    for i in range(chroma.shape[0]):
                        feature_dict[f"Chroma_{i+1}"] = np.mean(chroma[i])

                    # Spectral Contrast
                    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                    for i in range(contrast.shape[0]):
                        feature_dict[f"Contrast_{i+1}"] = np.mean(contrast[i])

                    # Tonnetz
                    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
                    for i in range(tonnetz.shape[0]):
                        feature_dict[f"Tonnetz_{i+1}"] = np.mean(tonnetz[i])

                    # Tempo
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    feature_dict["Tempo"] = tempo

                    features_list.append(feature_dict)

                except Exception as e:
                    print(f"⚠️ Error procesando {file_path}: {e}")

# Convertir a DataFrame y guardar
df = pd.DataFrame(features_list)
df.to_csv("dataset_caracteristicas_avanzadas.csv", index=False)
print("✅ CSV guardado: dataset_caracteristicas_avanzadas.csv")
