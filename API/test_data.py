# Definir sujetos ficticios como ejemplo para las predicciones del nivel de riesgo

sujeto1 = [
    'Cada mes', 
    'Varias veces al mes', 
    'Fines recreativos', 
    'Ambos', 
    'No', 
    'No', 
    'Si', 
    'No', 
    'Más de tres', 
    'Microdosis', 
    'N/A', 
    4, 
    'Trastorno Depresivo Mayor o Persistente;Enfermedad de Alzheimer', 
    'Trastorno Depresivo Mayor o Persistente;Trastorno de Ansiedad Generalizada (TAG)', 
    'Reducción de ansiedad', 
    'No tuvo ningún efecto negativo', 
    'Reducción de ansiedad;Reducción de Sintomas de depresión;Aumento de introspección;Mayor sentido de propósito o satisfacción con la vida;Mejora del sueño', 
    'No tuvo ningún efecto negativo'
]

sujeto2 = [
    'Varias veces al año', 
    'Varias veces al año', 
    'Fines terapéuticos', 
    'Fines terapéuticos', 
    'No', 
    'No', 
    'No', 
    'No', 
    'Dos', 
    'Macrodosis', 
    '1-5 sesiones de un día', 
    4, 
    'Trastorno Bipolar;Paranoia', 
    'Esquizofrenia;Psicosis;Adicción a las sustancias sintéticas o drogas ilegales;Adicción al alcohol', 
    'Aumento de creatividad;Aumento de apetito;Mejora del sueño', 
    'Problemas de memoria o atención;Psicosis;Problemas respiratorios;Trastornos del sueño', 
    'Aumento de introspección;Mayor sentido de propósito o satisfacción con la vida;Aumento de apetito', 
    'Cambios de humor'
]

sujeto3 = [
    'Diario', 
    'Varias veces al mes', 
    'Ambos', 
    'Fines recreativos', 
    'Si', 
    'No', 
    'Si', 
    'No', 
    'N/A', 
    'Macrodosis', 
    'N/A', 
    5, 
    'Trastorno Depresivo Mayor o Persistente;Trastorno de Ansiedad Generalizada (TAG);Adicción a juegos o apuestas', 
    'No sufro de ninguna condición relevante', 
    'Aumento de apetito;Mejora del sueño;Reducción de ansiedad', 
    'No tuvo ningún efecto negativo', 
    'Aumento de introspección;Mayor sentido de propósito o satisfacción con la vida', 
    'No tuvo ningún efecto negativo'
]

sujeto4 = [
    'Varias veces al mes', 
    'Varias veces al año', 
    'Fines recreativos', 
    'Ambos', 
    'No', 
    'No', 
    'No', 
    'No', 
    'Uno', 
    'Microdosis', 
    'N/A', 
    5, 
    'Enfermedad de Alzheimer;Adicción a la nicotina', 
    'Trastorno de Ansiedad Generalizada (TAG);Adicción a la nicotina', 
    'Mejora en el estado de ánimo', 
    'Problemas de memoria o atención', 
    'Reducción de ansiedad', 
    'No tuvo ningún efecto negativo'
]

sujeto5 = [
    'Varias veces al mes', 
    'Varias veces al año', 
    'Fines terapéuticos', 
    'Ambos', 
    'No', 
    'No', 
    'No', 
    'No', 
    'Más de tres', 
    'Macrodosis', 
    '1-5 sesiones de un día', 
    3, 
    'Trastorno Depresivo Mayor o Persistente;Trastorno Bipolar;Enfermedad de Alzheimer;Adicción a la nicotina;Adicción a las sustancias sintéticas o drogas ilegales;Adicción al alcohol', 
    'Adicción a las sustancias sintéticas o drogas ilegales;Adicción al alcohol', 
    'No tuvo ningún efecto positivo', 
    'Problemas de memoria o atención;Psicosis;Problemas respiratorios', 
    'Mayor sentido de propósito o satisfacción con la vida', 
    'Problemas de memoria o atención;Psicosis;Otros'
]

# Riesgo Alto
sujeto6 = [
    'Varias veces a la semana', 
    'Cada mes', 
    'Fines recreativos', 
    'Fines recreativos', 
    'Si', 
    'Si', 
    'Si', 
    'Si', 
    'N/A', 
    'Macrodosis', 
    'N/A', 
    'N/A', 
    'Trastorno Depresivo Mayor o Persistente;Psicosis', 
    'Trastorno Bipolar', 
    'Mejora en la introspección / conexión con el ser;Aumento de creatividad;Mejora en el estado de ánimo;Alivio de dolores crónicos;Aumento de apetito;Mejora del sueño;Reducción de inflamación o espasmos;Reducción de ansiedad', 
    'Problemas de memoria o atención;Aislamiento;Psicosis;Falta de apetito;Trastornos del sueño;Otros', 
    'Aumento de introspección;Mejora del sueño', 
    'Psicosis'
]

# Riesgo Bajo
sujeto7 = [
    'Varias veces a la semana', 
    'Cada año', 
    'Fines recreativos', 
    'Fines terapéuticos', 
    'No', 
    'No', 
    'No', 
    'No', 
    'Más de tres', 
    'Macrodosis', 
    '1-5 sesiones de un día', 
    5, 
    'No hay condiciones relevantes en mi familia', 
    'No sufro de ninguna condición relevante', 
    'Mejora en la introspección / conexión con el ser;Aumento de creatividad;Mejora en el estado de ánimo', 
    'No tuvo ningún efecto negativo', 
    'Aumento de introspección;Mayor sentido de propósito o satisfacción con la vida;Mejora del sueño', 
    'No tuvo ningún efecto negativo'
]

sujeto8 = [
    'Varias veces a la semana', 
    'Varias veces al año', 
    'Ambos', 
    'Fines terapéuticos', 
    'No', 
    'No', 
    'No', 
    'No', 
    'Dos', 
    'Microdosis', 
    'N/A', 
    4, 
    'Trastorno Depresivo Mayor o Persistente;Trastorno de Ansiedad Generalizada (TAG);Otros', 
    'Trastorno Depresivo Mayor o Persistente;Trastorno de Ansiedad Generalizada (TAG)', 
    'Mejora en la introspección / conexión con el ser;Mejora del sueño;Reducción de ansiedad;Otros', 
    'Problemas de memoria o atención;Aislamiento', 
    'Reducción de ansiedad;Reducción de Sintomas de depresión;Mayor sentido de propósito o satisfacción con la vida', 
    'Problemas de memoria o atención'
]

sujeto9 = [
    'Diario', 
    'Cada año', 
    'Ambos', 
    'Fines terapéuticos', 
    'Si', 
    'No', 
    'Si', 
    'No', 
    'Más de tres', 
    'Macrodosis', 
    '1-5 sesiones de un día', 
    5, 
    'Trastorno de Ansiedad Generalizada (TAG);Enfermedad de Alzheimer;Adicción al alcohol;Otros', 
    'Trastorno Depresivo Mayor o Persistente', 
    'Mejora en la introspección / conexión con el ser;Aumento de creatividad;Mejora en el estado de ánimo;Aumento de apetito;Mejora del sueño;Reducción de inflamación o espasmos', 
    'Aislamiento;Falta de apetito', 
    'Aumento de introspección;Mejora del sueño', 
    'No tuvo ningún efecto negativo'
]

sujeto10 = [
    'Varias veces al mes', 
    'Varias veces al año', 
    'Fines terapéuticos', 
    'Ambos', 
    'No', 
    'No', 
    'No', 
    'No', 
    'Uno', 
    'Macrodosis', 
    'Una sesión de un día', 
    3, 
    'Trastorno Depresivo Mayor o Persistente;Trastorno de Ansiedad Generalizada (TAG);Adicción a la nicotina;Adicción al alcohol', 
    'Trastorno esquizoafectivo;Paranoia;Psicosis;Adicción a las sustancias sintéticas o drogas ilegales', 
    'Mejora en la introspección / conexión con el ser;Mejora en el estado de ánimo;Reducción de ansiedad',
    'Problemas de memoria o atención;Psicosis',
    'No tuvo ningún efecto positivo',
    'Psicosis'
]
