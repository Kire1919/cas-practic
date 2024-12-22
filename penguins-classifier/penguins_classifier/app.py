# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar los modelos entrenados
logreg_model = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
knn_model = joblib.load('knn_model.pkl')

# Ruta para predecir con Regresión Logística
@app.route('/predict/logreg', methods=['POST'])
def predict_logreg():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = logreg_model.predict(features)
    return jsonify({'prediction': prediction[0]})

# Ruta para predecir con SVM
@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = svm_model.predict(features)
    return jsonify({'prediction': prediction[0]})

# Ruta para predecir con Árbol de Decisión
@app.route('/predict/dt', methods=['POST'])
def predict_dt():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = dt_model.predict(features)
    return jsonify({'prediction': prediction[0]})

# Ruta para predecir con KNN
@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = knn_model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
