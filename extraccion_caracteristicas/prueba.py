import librosa
import numpy as np
import pandas as pd

# Cargar archivo de audio
audio_path = './audios_colmena/100m.wav'
y, sr = librosa.load(audio_path, sr=None)

# MFCCs
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1)

# Zero-crossing rate
zcr = librosa.feature.zero_crossing_rate(y)[0]
zcr_mean = np.mean(zcr)

# Espectrograma
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(abs(S))

# Chroma
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_mean = np.mean(chroma, axis=1)

# Spectral Centroid
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
centroid_mean = np.mean(centroid)

# Agregar todo a un vector
caracteristicas = np.concatenate([mfcc_mean, [zcr_mean], chroma_mean, [centroid_mean]])

# Crear un DataFrame de Pandas y guardar en CSV
df = pd.DataFrame(caracteristicas.reshape(1, -1), 
                  columns=[f'MFCC_{i}' for i in range(13)] + ['ZCR'] + [f'Chroma_{i}' for i in range(12)] + ['Centroid'])
df.to_csv('./extraccion_caracteristicas/caracteristicas_audio.csv', index=False)
