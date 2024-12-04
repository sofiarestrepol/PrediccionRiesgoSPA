import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from expert_system import *
from definitions import *


def preprocess_data(df):
    # Verificar si hay valores nulos y en caso tal reemplazarlos por 'Sin Dato'
    if df.isnull().values.any():
        df.fillna('Sin Dato', inplace=True)
    
    df.replace({'N/A': 'Sin Dato'}, inplace=True)



def get_one_hot_encoding(df):
    # Separar las opciones de respuesta para cada una de las columnas categóricas y generar una lista con las respuestas
    for col in columnas_categoricas:
        df[col] = df[col].str.split(';')  
    # Crear columnas binarias usando 'explode' para descomponer las listas en filas
    for col in columnas_categoricas:
        df = df.explode(col)
    # Aplicar One-Hot Encoding a las columnas categóricas del DF con los datos explotados
    df_encoded = pd.get_dummies(df, columns=columnas_categoricas, prefix=columnas_categoricas)
    # Agrupar por el índice original para reconstruir el DataFrame en la forma deseada
    df_encoded = df_encoded.groupby(df_encoded.index).max().reset_index(drop=True)

    df = df_encoded.copy()

    return df_encoded, df

    

def create_col_psicosis_paranoia(df):
    # Verificar si las columnas existen en el DataFrame
    cols_existentes = [col for col in ['Condición_Psicosis', 'Condición_Paranoia', 'Historial Familiar_Psicosis', 'Historial Familiar_Paranoia'] if col in df.columns]

    if cols_existentes:
        # Crear la nueva columna combinada basada en las columnas existentes
        df['Condición_Psicosis/Paranoia'] = df[cols_existentes].any(axis=1)
        # Eliminar las columnas antiguas
        df.drop(cols_existentes, axis=1, inplace=True)


def transform_to_bool(df):
    # Renombrar columnas en texto a valores binarios
    for col in cols_dependencia_abuso:
        df[col] = df[col].map(dict_cols_binarias)

    # Convertir las columnas binarias (1,0) en booleanas (True, False)
    for col in cols_dependencia_abuso:
        df[col] = df[col].astype(bool)



def rename_cols(df):
    # Renombrar las columnas para tener mas claridad y facil acceso
    df.rename(columns=dict_renombrar_respuestas, inplace=True)



def get_label_encoding(df_test_encoded):
    # Realizar codificación con Label Encoding para variables seleccionadas
    cols_label_encoder = [col for col in df_test_encoded.columns if 'Frecuencia' in col]

    # Codificación Frecuencia de Consumo
    for col in cols_label_encoder:
        df_test_encoded[col] = df_test_encoded[col].map(dict_encoder_frecuencia)
        df_test_encoded[col] = df_test_encoded[col].astype(int)

    # Codificación Cantidad de Sesiones con Macrodosis
    df_test_encoded['Sesiones Macrodosis'] = df_test_encoded['Sesiones Macrodosis'].map(dict_encoder_sesiones_macro)
    df_test_encoded['Sesiones Macrodosis'] = df_test_encoded['Sesiones Macrodosis'].astype(int)

    # Codificación Cantidad de Tratamientos con SPA
    df_test_encoded['Cantidad Tratamientos'] = df_test_encoded['Cantidad Tratamientos'].map(dict_encoder_cantidad_tratamientos)
    df_test_encoded['Cantidad Tratamientos'] = df_test_encoded['Cantidad Tratamientos'].astype(int)





def transform_data(df):
    try:
        # Crear variable fusionada (Psicosis/Paranoia)
        create_col_psicosis_paranoia(df)
        # Transformar los datos a valores booleanos
        transform_to_bool(df)
        # Renombrar columnas necesarias
        rename_cols(df)
        
        # Reemplazar caracteres especiales en los nombres de las columnas
        df.columns = df.columns.str.replace(r'[^\w\s/,\']', '_', regex=True)


    except Exception as e:
        print(f'Ocurrió un error en el preprocesamiento de los datos: {e}')

    else:
        return df




def divide_dataset(df_test_encoded):
    # Generar un DF para cada sustancia
    df_test_encoded_cannabis = df_test_encoded.copy()
    df_test_encoded_psilocibina = df_test_encoded.copy()

    for col in df_test_encoded_cannabis.columns:
        if 'Psilocibina' in col or 'Otros' in col or 'Sin Dato' in col or 'Tipo de Dosis' in col or 'Sin Razón' in col:
            df_test_encoded_cannabis.drop(columns=[col], inplace=True)


    for col in df_test_encoded_psilocibina.columns:
        if 'Cannabis' in col or 'Otros' in col or 'Sin Dato' in col or 'Sin Razón' in col:
            df_test_encoded_psilocibina.drop(columns=[col], inplace=True)

    return df_test_encoded_cannabis, df_test_encoded_psilocibina


def execute_expert_system(df_test, df_test_encoded, target_col):
    try: 
        # Definir el conjunto de reglas según la sustancia
        if 'Cannabis' in target_col:
            riesgo_bajo = get_low_risk_cannabis(df_test)
            riesgo_medio = get_medium_risk_cannabis(df_test)
            riesgo_alto = get_high_risk_cannabis(df_test)

        elif 'Psilocibina' in target_col:
            riesgo_bajo = get_low_risk_psilocibina(df_test)
            riesgo_medio = get_medium_risk_psilocibina(df_test)
            riesgo_alto = get_high_risk_psilocibina(df_test)

        # Inicializar la nueva variable con 'Riesgo Desconocido'
        df_test_encoded[target_col] = 'Riesgo Desconocido'
        df_test[target_col] = 'Riesgo Desconocido'

        # Asignar un nivel de riesgo bajo a los casos que lo cumplan
        df_test_encoded.loc[riesgo_bajo, target_col] = 'Riesgo Bajo'
        df_test.loc[riesgo_bajo, target_col] = 'Riesgo Bajo'

        # Asignar el nivel de riesgo medio a los casos que lo cumplan y que no tengan un valor de riesgo asociado
        df_test_encoded.loc[(df_test_encoded[target_col] == 'Riesgo Desconocido') & riesgo_medio, 
                    target_col] = 'Riesgo Medio'
        df_test.loc[(df_test[target_col] == 'Riesgo Desconocido') & riesgo_medio, 
                    target_col] = 'Riesgo Medio'

        # Se añade el nivel de riesgo alto a los casos que lo cumplan y que no tengan un valor de riesgo asociado
        df_test_encoded.loc[(df_test_encoded[target_col] == 'Riesgo Desconocido') & riesgo_alto, 
                    target_col] = 'Riesgo Alto'
        df_test.loc[(df_test[target_col] == 'Riesgo Desconocido') & riesgo_alto, 
                    target_col] = 'Riesgo Alto'
        
    except Exception as e:
        print(f'Ocurrió un error en la ejecución del sistema experto: {e}')



def encode_risk_level(df_test_encoded, target_col):
    # Codificación del Nivel de Riesgo del Tratamiento 
    df_test_encoded[target_col] = df_test_encoded[target_col].map(dict_encoder_riesgo_tratamiento)



def filter_df(df_encoded, target_col):
    # Eliminar filas donde la clase objetivo tiene solo un miembro
    df_encoded_filtrado = df_encoded[df_encoded[target_col] != 0]
    return df_encoded_filtrado



def setup_training_data(df_encoded, target_col, random_state):
    df_encoded_filtrado = df_encoded[df_encoded[target_col] != 0]
    # Definir la variable objetivo (y) y las caracteristicas (X)
    X_riesgo = df_encoded_filtrado.drop(columns=[target_col])
    y_riesgo = df_encoded_filtrado[target_col]

    # Eliminar las clases con menos de dos elementos
    counts = y_riesgo.value_counts()
    valid_classes = counts[counts >= 2].index
    X_riesgo = X_riesgo[y_riesgo.isin(valid_classes)]
    y_riesgo = y_riesgo[y_riesgo.isin(valid_classes)]

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X_train_riesgo, X_test_riesgo, y_train_riesgo, y_test_riesgo = train_test_split(
        X_riesgo, 
        y_riesgo, 
        test_size=0.20,
        stratify=y_riesgo,
        random_state=random_state)
    
    return X_riesgo, y_riesgo, X_train_riesgo, X_test_riesgo, y_train_riesgo, y_test_riesgo



def setup_test_data(df_test_encoded, X_test_riesgo, target_col):
    # Definir la variable objetivo
    y_test_riesgo = df_test_encoded[target_col]

    # Re ordenar el DF de prueba para que tenga el mismo formato que los datos de entrenamiento
    df_test_encoded_model = df_test_encoded.reindex(columns=X_test_riesgo.columns)
    
    return df_test_encoded_model, y_test_riesgo

def balance_and_setup_test_data(df_encoded, df_test_encoded, target_col, random_state):
    # Definir los subconjuntos de entranmiento y prueba para los datos de entrenamiento
    X_riesgo, y_riesgo, X_train_riesgo, X_test_riesgo, y_train_riesgo, y_test_riesgo = setup_training_data(df_encoded, target_col, random_state)

    # Convertir los nombres de las columnas en conjuntos
    columnas_X_riesgo = set(X_riesgo.columns)
    columnas_df_test_encoded = set(df_test_encoded.columns)

    # Identificar las columnas faltantes en los datos de prueba
    columnas_faltantes = columnas_X_riesgo - columnas_df_test_encoded

    # Crear las columnas faltantes en los datos de prueba con valores False
    for columna in columnas_faltantes:
        df_test_encoded[columna] = False


    # Re ordenar el DF de prueba para que tenga el mismo formato que los datos de entrenamiento
    df_test_encoded_model, y_test_riesgo = setup_test_data(df_test_encoded, X_test_riesgo, target_col)

    return df_test_encoded_model, X_riesgo, y_riesgo, X_train_riesgo, X_test_riesgo, y_train_riesgo, y_test_riesgo 


# Definir una función de mapeo para convertir el nivel de riesgo codificado a lenguaje natural
reverse_dict_encoder_riesgo_tratamiento = {v: k for k, v in dict_encoder_riesgo_tratamiento.items()}
map_values = np.vectorize(lambda x: reverse_dict_encoder_riesgo_tratamiento.get(x, None))



# def train_model(X_train, y_train, model_name, parameters):
#     if model_name == 'Gradient Boosting Classifier':
#         model = GradientBoostingClassifier(**parameters)

#     model.fit(X_train, y_train)

#     return model
    


