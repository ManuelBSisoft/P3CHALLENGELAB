
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Cargar el dataset
datos = pd.read_csv("sensores_planta.csv")

# 2. Limpiar registros vacios
datos = datos.dropna()

# 3. Separar variables de entrada y variable objetivo
X = datos.drop("estado", axis=1)
y = datos["estado"]

# 4. Dividir los datos: 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Crear el modelo
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 6. Entrenar el modelo
modelo.fit(X_train, y_train)

# 7. Realizar predicciones
y_pred = modelo.predict(X_test)

# 8. Calcular precision
precision = accuracy_score(y_test, y_pred)

print("--------------------------------")
print(" SISTEMA DE MANTENIMIENTO")
print("--------------------------------")
print(f"Registros utilizados: {len(datos)}")
print(f"Datos de entrenamiento: {len(X_train)}")
print(f"Datos de prueba: {len(X_test)}")
print(f"Accuracy: {precision * 100:.2f}%")

print("\nReporte de clasificacion:")
print(classification_report(y_test, y_pred))

# 9. Guardar el modelo entrenado
joblib.dump(modelo, "modelo_mantenimiento.pkl")

print("\nModelo guardado correctamente.")
