import requests
import json

# Características de entrada para la predicción (dos ejemplos diferentes)
features_1 = [45.0, 13.0, 195.0, 4050.0, 1, 0, 0]  # Ejemplo 1
features_2 = [40.0, 12.0, 180.0, 3700.0, 0, 1, 0]  # Ejemplo 2

# URL del servidor Flask
url_logreg = 'http://127.0.0.1:5000/predict/logreg'
url_svm = 'http://127.0.0.1:5000/predict/svm'
url_dt = 'http://127.0.0.1:5000/predict/dt'
url_knn = 'http://127.0.0.1:5000/predict/knn'

# Función para enviar solicitud POST
def predict_model(url, features):
    response = requests.post(url, json={'features': features})
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"

# Realizar las predicciones con todos los modelos (dos peticiones por modelo)
response_logreg_1 = predict_model(url_logreg, features_1)
response_logreg_2 = predict_model(url_logreg, features_2)

response_svm_1 = predict_model(url_svm, features_1)
response_svm_2 = predict_model(url_svm, features_2)

response_dt_1 = predict_model(url_dt, features_1)
response_dt_2 = predict_model(url_dt, features_2)

response_knn_1 = predict_model(url_knn, features_1)
response_knn_2 = predict_model(url_knn, features_2)

# Mostrar las predicciones
print(f"Predicción Regresión Logística (1): {response_logreg_1}")
print(f"Predicción Regresión Logística (2): {response_logreg_2}")
print(f"Predicción SVM (1): {response_svm_1}")
print(f"Predicción SVM (2): {response_svm_2}")
print(f"Predicción Árbol de Decisión (1): {response_dt_1}")
print(f"Predicción Árbol de Decisión (2): {response_dt_2}")
print(f"Predicción KNN (1): {response_knn_1}")
print(f"Predicción KNN (2): {response_knn_2}")
