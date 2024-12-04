# Importaciones
import pandas as pd
import numpy as np
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException
from joblib import load
from dotenv import load_dotenv
import os

from expert_system import * 
from utils import *
from test_data import *
from definitions import columnas_df
 

# Inicializar app
app = FastAPI()

# Leer datos de entrenamiento
df_encoded_cannabis = pd.read_csv('../encuestas/cannabis_encoded_modelos.csv')
df_encoded_psilocibina = pd.read_csv('../encuestas/psilocibina_encoded_modelos.csv')


# Cargar y extraer variables de entorno
load_dotenv()
random_state_cannabis = int(os.getenv("RANDOM_STATE_CANNABIS"))
random_state_psilocibina = int(os.getenv("RANDOM_STATE_PSILOCIBINA"))
target_col_cannabis = os.getenv("TARGET_COL_CANNABIS")
target_col_psilocibina = os.getenv("TARGET_COL_PSILOCIBINA")


try:
    model_psilocibina = load('../modelos/best_model_psilocibina.joblib')
    model_cannabis = load('../modelos/best_model_cannabis.joblib')
    print(f'Los modelos se cargaron correctamente')
except Exception as e:
    print(f'Ocurrió un error en la carga de los modelos: {e}')


# Definir el formato de los datos a predecir
class DataPredict(BaseModel):
    data_to_predict: list[list] = [sujeto7]


@app.post("/predict-risk")
def predict(request: DataPredict):
    """
    Predice el nivel de riesgo para un tratamiento con sustancias psicoactivas según el perfil del paciente.

    - Frecuencia de Consumo de Psilocibina
    - Frecuencia de Consumo de Cannabis
    - Propósito del Consumo de Cannabis
    - Propósito del Consumo de Psilocibina
    - Dependencia al Cannabis
    - Dependencia a la Psilocibina
    - Consumo Abusivo de Cannabis
    - Consumo Abusivo de Psilocibina
    - Cantidad de Tratamientos con Psilocibina
    - Tipo de Dosis de Psilocibina
    - Cantidad de Sesiones de Macrodosis con Psilocibina
    - Calificación del Tratamiento
    - Historial Familiar
    - Condiciones Medicas
    - Efectos Positivos Cannabis
    - Efectos Negativos Cannabis
    - Efectos Positivos Psilocibina
    - Efectos Negativos Psilocibina
    
    Las preguntas de 'Frecuencia de Consumo' permiten las siguientes opciones de respuesta: 
    - Diario, Varias veces a la semana, Cada semana, Varias veces al mes, Cada mes, Varias veces al año, Cada año.

    Las preguntas de 'Propósito del Consumo' determinan con qué finalidad se consumió la sustancia, y permiten las siguientes opciones de respuesta:
    - Fines recreativos, Fines terapéuticos, Ambos.

    Las preguntas de 'Dependencia' y 'Consumo Abusivo' definen si el paciente ha experimentado dependencia a la sustancia y si ha abusado de ella respectivamente, y sus respuestas tienen formato de 'Si' o 'No'.

    La 'Cantidad de Tratamientos' se refiere a cuántos tratamientos con psilocibina ha realizado el paciente previamente. Las opciones de respuesta validas son:
    - Uno, Dos, Más de tres, y N/A en caso de no haber realizado tratamientos previos.

    'Tipo de Dosis' representa la dosificación utilizada para el consumo de psilocibina del paciente. Permite las siguientes opciones:
    - Microdosis, Macrodosis, Ambas.

    La pregunta 'Cantidad de Sesiones de Macrodosis' hace referencia a cuántas sesiones de macrodosis ha realizado el paciente con psilocibina. Las posibles opciones de respuesta son:
    - Una sesión de un día, 1-5 sesiones de un día, Más de 10 sesiones de un día, Otros
    
    La 'Calificación del Tratamiento' define un valor entre 1 y 5 que el paciente otorga a los tratamientos previos que haya realizado con alguna de las sustancias psicoactivas.    

    Los campos 'Condiciones Medicas' e 'Historial Familiar' lista los trastornos significativos que presente el participante y aquellos que presente un miembro de su familia respectivamente. 

    Las preguntas de 'Efectos Positivos' y 'Efectos Negativos' listan los efectos que haya experimentado el paciente con cada una de las sustancias.
        
    Todas los campos permiten la opción 'N/A' como respuesta en caso de que la pregunta no aplique para el paciente.

    """
    try:
        # Obtener los datos de prueba recibidos en la solicitud a la API y convertirlos en un DataFrame
        list_data = request.data_to_predict
        df_test = pd.DataFrame(list_data, columns=columnas_df)

        # Realizar el preprocesamiento de los datos de prueba
        preprocess_data(df_test)

        # Codificar las variables con multiples respuestas
        df_test_encoded, df_test = get_one_hot_encoding(df_test)
        
        # Realizar transformaciones necesarias a los datos de prueba
        df_test = transform_data(df_test)
        df_test_encoded = transform_data(df_test_encoded)

        # Codificar con Label Encoding las variables con una gran cantidad de posibilidades de respuesta
        get_label_encoding(df_test_encoded)
        # Condificar con One Hot Encoding el resto de variables
        df_test_encoded = pd.get_dummies(df_test_encoded) 

        # Dividir el dataset de prueba según la sustancia
        df_test_encoded_cannabis, df_test_encoded_psilocibina = divide_dataset(df_test_encoded)

        # Ejecutar el sistema experto con los conjuntos de reglas para determinar el nivel de riesgo del individuo
        execute_expert_system(df_test, df_test_encoded_cannabis, target_col_cannabis)
        execute_expert_system(df_test, df_test_encoded_psilocibina, target_col_psilocibina)

        # Codificar el nivel de riesgo
        encode_risk_level(df_test_encoded_cannabis, target_col_cannabis)
        encode_risk_level(df_test_encoded_psilocibina, target_col_psilocibina)

        # Filtrar los datos de prueba para eliminar filas sin predicciones de riesgo
        df_test_encoded_cannabis = filter_df(df_test_encoded_cannabis, target_col_cannabis)
        df_test_encoded_psilocibina = filter_df(df_test_encoded_psilocibina, target_col_psilocibina)


        # Generar el DF para el modelo y sus subconjuntos de entrenamiento y prueba
        df_test_encoded_cannabis_model, _ , _ , _ , _ ,_ , _  = balance_and_setup_test_data(df_encoded_cannabis, df_test_encoded_cannabis, target_col_cannabis, random_state_cannabis)
        df_test_encoded_psilocibina_model, _ , _ , _ , _ , _ , _  = balance_and_setup_test_data(df_encoded_psilocibina, df_test_encoded_psilocibina, target_col_psilocibina, random_state_psilocibina)

        # Ejecutar el modelo pre cargado para realizar predicciones para ambas sustancias
        if not df_test_encoded_psilocibina_model.empty:
            y_test_pred_riesgo_psilocibina = model_psilocibina.predict(df_test_encoded_psilocibina_model)
        else:
            y_test_pred_riesgo_psilocibina = np.array([0])

        if not df_test_encoded_cannabis_model.empty:
            y_test_pred_riesgo_cannabis = model_cannabis.predict(df_test_encoded_cannabis_model)
        else:
            y_test_pred_riesgo_cannabis = np.array([0])




        # Reemplazar los valores codificados para obtener el nivel de riesgo en lenguaje natural
        y_test_pred_riesgo_cannabis = map_values(y_test_pred_riesgo_cannabis)
        y_test_pred_riesgo_psilocibina = map_values(y_test_pred_riesgo_psilocibina)

        return {
            "Riesgo Cannabis": {
                "Predicción Sistema Experto": df_test['Nivel de Riesgo Tratamiento Cannabis'][0],
                "Predicción Modelo Gradient Boosting": str(y_test_pred_riesgo_cannabis[0])
            },
            "Riesgo Psilocibina": {
                "Predicción Sistema Experto": df_test['Nivel de Riesgo Tratamiento Psilocibina'][0],
                "Predicción Modelo Gradient Boosting": str(y_test_pred_riesgo_psilocibina[0])
            }
        }
    except Exception as e:
        print(f'Exception: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {'Proyecto de Fin de Programa - SRL'}
