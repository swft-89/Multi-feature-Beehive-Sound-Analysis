from pydub import AudioSegment
import os

AUDIO_PATH = "audios_colmena/500m.wav"
SEGMENT_DIR = "audios_colmena/audios_segmentados_500m"
SEGMENT_DURATION = 10 * 1000  # 10 segundos en milisegundos

# Cargar audio
audio = AudioSegment.from_wav(AUDIO_PATH)
total_length = len(audio)
os.makedirs(SEGMENT_DIR, exist_ok=True)

# Segmentar
count = 0
for i in range(0, total_length, SEGMENT_DURATION):
    segment = audio[i:i + SEGMENT_DURATION]
    segment.export(os.path.join(SEGMENT_DIR, f"segmento_{count}.wav"), format="wav")
    count += 1

print(f"✅ Segmentación completa: {count} fragmentos guardados en '{SEGMENT_DIR}'")
