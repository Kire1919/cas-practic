# model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from data_preparation import preprocess_data, load_data, clean_data

# Entrenar y evaluar modelos
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Modelo {name} entrenado con éxito")
        
        # Evaluar modelo
        score = model.score(X_test, y_test)
        print(f"Precisión del modelo {name}: {score}")
        
        # Serializar el modelo
        joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")
        print(f"Modelo {name} guardado como {name.lower().replace(' ', '_')}_model.pkl")

# Función principal
def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    train_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
