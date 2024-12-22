# data_preparation.py

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Cargar el dataset
def load_data():
    df = sns.load_dataset('penguins')  # Carga el dataset de Seaborn
    return df

# Limpiar el dataset (eliminar filas con valores NA)
def clean_data(df):
    df = df.dropna()  # Eliminar las filas con valores NA
    return df

# Preprocesamiento de los datos
def preprocess_data(df):
    # Separar las características y la etiqueta
    X = df.drop('species', axis=1)  # Características
    y = df['species']  # Etiqueta (especie del pingüino)

    # Codificación one-hot para variables categóricas
    categorical_features = ['island', 'sex']
    numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    # Crear transformadores para preprocesar las características
    categorical_transformer = OneHotEncoder(drop='first')  # Codificación one-hot
    numerical_transformer = StandardScaler()  # Normalización de variables numéricas

    # Crear un pipeline de procesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Crear un pipeline completo que incluya el preprocesador
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Aplicar la transformación a las características
    X_processed = pipeline.fit_transform(X)
    
    # Dividir en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Función principal
def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("Datos preparados con éxito")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()

