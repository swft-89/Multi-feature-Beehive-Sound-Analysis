# Unir los CSVs
import pandas as pd

df_100 = pd.read_csv('./extraccion_caracteristicas/resultados/audio_segmentado100m_normalizado.csv')
df_100['Etiqueta'] = '100m'

df_300 = pd.read_csv('./extraccion_caracteristicas/resultados/audio_segmentado300m_normalizado.csv')
df_300['Etiqueta'] = '300m'

df_500 = pd.read_csv('./extraccion_caracteristicas/resultados/audio_segmentado500m_normalizado.csv')
df_500['Etiqueta'] = '500m'

df = pd.concat([df_100, df_300, df_500], ignore_index=True)
df.to_csv('./extraccion_caracteristicas/resultados/dataset_clasificacion_pecoreo.csv', index=False)
