# entrenamiento.py - Versión con estructura PSO-LSTM para audio de abejas

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pyswarms.single.global_best import GlobalBestPSO
import matplotlib.pyplot as plt
import os

# --- Cargar y preparar datos de audio ---
def cargar_datos():
    # Cargar datasets (ejemplo con archivos existentes)
    X = pd.read_csv("./reduccion_caracteristicas/resultados/500m/pso_reducido_500m.csv").drop(columns=["Segmento"])
    
    # Si no hay etiquetas reales, crear unas simuladas binarias
    y = np.random.randint(0, 2, size=X.shape[0])  # Reemplazar con tus etiquetas reales
    
    # Normalización
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, y, scaler

# --- Crear secuencias para LSTM ---
def crear_secuencias(data, labels, look_back=10):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:i+look_back])
        y.append(labels[i+look_back])
    return np.array(X), np.array(y)

# --- Función objetivo para PSO ---
def funcion_objetivo(params, X_train, y_train, look_back):
    costs = np.zeros(params.shape[0])
    
    for i in range(params.shape[0]):
        n_units = int(params[i, 0])
        batch_size = int(params[i, 1])
        
        # Modelo LSTM
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(look_back, X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=0)
        
        costs[i] = history.history['loss'][-1]
    
    return costs

# --- Entrenar modelo final ---
def entrenar_modelo_final(X_train, y_train, X_test, y_test, best_params, look_back):
    n_units = int(best_params[0])
    batch_size = int(best_params[1])
    
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(look_back, X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=30, batch_size=batch_size, validation_data=(X_test, y_test))
    
    return model, history

# --- Evaluación y visualización ---
def evaluar_modelo(model, X_test, y_test, history):
    # Predicciones
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Métricas
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.show()
    
    # Gráfico de pérdida
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Pérdida durante el entrenamiento')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend()
    plt.show()

# --- Main ---
def main():
    # 1. Cargar y preparar datos
    X, y, scaler = cargar_datos()
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Crear secuencias LSTM
    look_back = 10  # Ajustar según necesidad
    X_train_seq, y_train_seq = crear_secuencias(X_train, y_train, look_back)
    X_test_seq, y_test_seq = crear_secuencias(X_test, y_test, look_back)
    
    # 4. Optimización con PSO
    print("🔍 Optimizando hiperparámetros con PSO...")
    bounds = (np.array([16, 1]), np.array([128, 24]))  # Límites para [n_units, batch_size]
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=5, dimensions=2, options=options, bounds=bounds)
    
    cost, best_params = optimizer.optimize(
        lambda params: funcion_objetivo(params, X_train_seq, y_train_seq, look_back), 
        iters=10
    )
    
    print(f"\n⚙️ Mejores hiperparámetros encontrados: {best_params}")
    
    # 5. Entrenamiento final
    print("\n🚀 Entrenando modelo final...")
    model, history = entrenar_modelo_final(
        X_train_seq, y_train_seq, 
        X_test_seq, y_test_seq, 
        best_params, look_back
    )
    
    # 6. Evaluación
    evaluar_modelo(model, X_test_seq, y_test_seq, history)

if __name__ == "__main__":
    import seaborn as sns  # Para la matriz de confusión
    main()
