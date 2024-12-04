# Variables originales del DF de prueba
columnas_df = ['Frecuencia Cannabis', 'Frecuencia Psilocibina', 'Propósito Cannabis', 'Propósito Psilocibina', 'Dependencia Cannabis', 'Dependencia Psilocibina', 'Abuso Cannabis', 'Abuso Psilocibina', 'Cantidad Tratamientos', 'Tipo de Dosis', 'Sesiones Macrodosis', 'Calificación Tratamiento', 'Historial Familiar', 'Condición', 'Efectos Positivos Cannabis', 'Efectos Negativos Cannabis', 'Efectos Positivos Psilocibina', 'Efectos Negativos Psilocibina']

# Lista de columnas categoricas con multiples respuestas para codificar con One Hot Encoding
columnas_categoricas = [
    'Historial Familiar',
    'Condición',
    'Efectos Positivos Cannabis', 
    'Efectos Negativos Cannabis', 
    'Efectos Positivos Psilocibina', 
    'Efectos Negativos Psilocibina' 
]

# Diccionario de mapeo para renombrar las variables
dict_renombrar_respuestas = {
    'Historial Familiar_Adicción a juegos o apuestas': 'Historial Familiar_Adicción Juegos o Apuestas',
    'Historial Familiar_Adicción a la nicotina': 'Historial Familiar_Adicción Nicotina',
    'Historial Familiar_Adicción a las sustancias sintéticas o drogas ilegales': 'Historial Familiar_Adicción Sustancias Sintenticas o Drogas Ilegales',
    'Historial Familiar_Adicción a medicamentos recetados': 'Historial Familiar_Adicción Medicamentos Recetados',
    'Historial Familiar_Adicción al alcohol': 'Historial Familiar_Adicción Alcohol',
    'Historial Familiar_Trastorno Bipolar (I, II)': 'Historial Familiar_Trastorno Bipolar',
    'Historial Familiar_No hay condiciones relevantes en mi familia': 'Historial Familiar_Sin Condición Relevante',
    'Condición_Adicción a juegos o apuestas': 'Condición_Adicción Juegos o Apuestas',
    'Condición_Adicción a la nicotina': 'Condición_Adicción Nicotina',
    'Condición_Adicción a las sustancias sintéticas o drogas ilegales': 'Condición_Adicción Sustancias Sintenticas o Drogas Ilegales',
    'Condición_Adicción a medicamentos recetados': 'Condición_Adicción Medicamentos Recetados',
    'Condición_Adicción al alcohol': 'Condición_Adicción Alcohol',
    'Condición_Trastorno Bipolar (I, II)': 'Condición_Trastorno Bipolar',
    'Condición_No sufro de ninguna condición relevante': 'Condición_Sin Condición Relevante',
    'Efectos Positivos Cannabis_Alivio de dolores crónicos': 'Efectos Positivos Cannabis_Alivio Dolores Crónicos',
    'Efectos Positivos Cannabis_Aumento de apetito': 'Efectos Positivos Cannabis_Aumento Apetito',
    'Efectos Positivos Cannabis_Aumento de creatividad': 'Efectos Positivos Cannabis_Aumento Creatividad',
    'Efectos Positivos Cannabis_Mejora del sueño': 'Efectos Positivos Cannabis_Mejora Sueño',
    'Efectos Positivos Cannabis_Mejora en el estado de animo': 'Efectos Positivos Cannabis_Mejora Estado de Animo',
    'Efectos Positivos Cannabis_Mejora en la introspección / conexión con el ser': 'Efectos Positivos Cannabis_Mejora Introspección',
    'Efectos Positivos Cannabis_No tuvo ningún efecto positivo': 'Efectos Positivos Cannabis_Sin Efecto Positivo',
    'Efectos Positivos Cannabis_Reducción de ansiedad': 'Efectos Positivos Cannabis_Reducción Ansiedad',
    'Efectos Positivos Cannabis_Reducción de inflamación o espasmos': 'Efectos Positivos Cannabis_Reducción Inflamacion o Espasmos',
    'Efectos Negativos Cannabis_Falta de apetito': 'Efectos Negativos Cannabis_Falta Apetito',
    'Efectos Negativos Cannabis_No tuvo ningún efecto negativo': 'Efectos Negativos Cannabis_Sin Efecto Negativo',
    'Efectos Negativos Cannabis_Problemas de memoria o atención': 'Efectos Negativos Cannabis_Problemas Memoria o Atención',
    'Efectos Positivos Psilocibina_Alivio de dolores crónicos': 'Efectos Positivos Psilocibina_Alivio Dolores Crónicos',
    'Efectos Positivos Psilocibina_Aumento de apetito': 'Efectos Positivos Psilocibina_Aumento Apetito',
    'Efectos Positivos Psilocibina_Aumento de introspección': 'Efectos Positivos Psilocibina_Mejora Introspección',
    'Efectos Positivos Psilocibina_Mayor sentido de propósito o satisfacción con la vida': 'Efectos Positivos Psilocibina_Mayor Satisfacción con la Vida',
    'Efectos Positivos Psilocibina_Mejora del sueño': 'Efectos Positivos Psilocibina_Mejora Sueño',
    'Efectos Positivos Psilocibina_No tuvo ningún efecto positivo': 'Efectos Positivos Psilocibina_Sin Efecto Positivo',
    'Efectos Positivos Psilocibina_Reducción de ansiedad': 'Efectos Positivos Psilocibina_Reducción Ansiedad',
    'Efectos Positivos Psilocibina_Reducción de sintomas de depresión': 'Efectos Positivos Psilocibina_Reducción Sintomas Depresión',
    'Efectos Negativos Psilocibina_No tuvo ningún efecto negativo': 'Efectos Negativos Psilocibina_Sin Efecto Negativo',
    'Efectos Negativos Psilocibina_Problemas de memoria o atención': 'Efectos Negativos Psilocibina_Problemas Memoria o Atención'
}

cols_dependencia_abuso = [
    'Dependencia Cannabis', 'Abuso Cannabis', 
    'Dependencia Psilocibina', 'Abuso Psilocibina'
]

dict_cols_binarias = {
    'Sin Dato': 0,
    'No': 0,
    'Si': 1
}

# Diccionarios de mapeo para codificar variables con opciones únicas
dict_encoder_frecuencia = {
    "Sin Dato": 0,
    "Diario": 1,
    "Varias veces a la semana": 2,
    "Cada semana": 3,
    "Varias veces al mes": 4,
    "Cada mes": 5,
    "Varias veces al año": 6,
    "Cada año": 7
}

dict_encoder_sesiones_macro = {
    "Sin Dato": 0,
    "Una sesión de un día": 1,
    "1-5 sesiones de un día": 2,
    "Más de 10 sesiones de un día": 3,
    "Otros": 4
}

dict_encoder_cantidad_tratamientos = {
    "Sin Dato": 0,
    "Uno": 1,
    "Dos": 2,
    "Más de tres": 3
}


# Diccionarios de mapeo para codificar el nivel de riesgo del tratamiento
dict_encoder_riesgo_tratamiento = {
    "Riesgo Desconocido": 0,
    "Riesgo Bajo": 1,
    "Riesgo Medio": 2,
    "Riesgo Alto": 3
}

dict_encoder_riesgo_tres_niveles = {
    "Riesgo Bajo": 1,
    "Riesgo Medio": 2,
    "Riesgo Alto": 3
}

