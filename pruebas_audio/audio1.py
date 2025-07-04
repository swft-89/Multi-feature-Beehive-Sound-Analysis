import librosa
import librosa.display
import matplotlib.pyplot as plt

# Cargar archivo de audio
audio_path = 'audios_colmena/100m.wav'
y, sr = librosa.load(audio_path, sr=None)

# Mostrar duración
print(f"Duración: {len(y)/sr:.2f} segundos")

# Mostrar forma de onda
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Forma de onda del audio")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()
