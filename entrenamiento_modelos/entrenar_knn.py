import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def cargar_datos(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["archivo", "etiqueta"])
    y = LabelEncoder().fit_transform(df["etiqueta"])
    return StandardScaler().fit_transform(X), y

def entrenar_y_mostrar_resultados(X, y, nombre_dataset):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(f"\n--- Resultados KNN para {nombre_dataset} ---")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

# Entrenamientos
X_ori, y_ori = cargar_datos("./dataset_caracteristicas_limpio.csv")
X_pso, y_pso = cargar_datos("./dataset_pso_reducido.csv")
X_cco, y_cco = cargar_datos("./dataset_cco_reducido.csv")

entrenar_y_mostrar_resultados(X_ori, y_ori, "Todas las características")
entrenar_y_mostrar_resultados(X_pso, y_pso, "PSO")
entrenar_y_mostrar_resultados(X_cco, y_cco, "CCO")
