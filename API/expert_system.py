import pandas as pd

# Variables relevantes para la determinación del nivel de riesgo
historial_familiar_condiciones_riesgosas = ['Historial Familiar_Esquizofrenia', 'Historial Familiar_Psicosis/Paranoia', 'Historial Familiar_Trastorno Bipolar']
condiciones_medicas_riesgosas = ['Condición_Esquizofrenia', 'Condición_Trastorno Bipolar', 'Condición_Psicosis/Paranoia']

historial_familiar_adicciones = ['Historial Familiar_Adicción Juegos o Apuestas','Historial Familiar_Adicción Nicotina','Historial Familiar_Adicción Sustancias Sintéticas o Drogas Ilegales','Historial Familiar_Adicción Medicamentos Recetados','Historial Familiar_Adicción Alcohol']
condiciones_medicas_adicciones = ['Condición_Adicción Juegos o Apuestas','Condición_Adicción Nicotina','Condición_Adicción Sustancias Sintéticas o Drogas Ilegales','Condición_Adicción Medicamentos Recetados','Condición_Adicción Alcohol']

efectos_positivos_cannabis = ['Efectos Positivos Cannabis_Aumento Apetito', 'Efectos Positivos Cannabis_Aumento Creatividad', 'Efectos Positivos Cannabis_Mejora Sueño', 'Efectos Positivos Cannabis_Mejora Estado de Animo', 'Efectos Positivos Cannabis_Mejora Introspección', 'Efectos Positivos Cannabis_Reducción Ansiedad', 'Efectos Negativos Cannabis_Sin Efecto Negativo']
efectos_moderados_cannabis = ['Efectos Negativos Cannabis_Aislamiento', 'Efectos Negativos Cannabis_Falta Apetito', 'Efectos Negativos Cannabis_Problemas Memoria o Atención', 'Efectos Negativos Cannabis_Trastornos del sueño', 'Efectos Negativos Cannabis_Problemas respiratorios', 'Efectos Negativos Cannabis_Problemas cognitivos']
efectos_negativos_determinantes_cannabis = ['Efectos Negativos Cannabis_Psicosis']

efectos_positivos_psilocibina = ['Efectos Positivos Psilocibina_Alivio Dolores Crónicos','Efectos Positivos Psilocibina_Aumento Apetito','Efectos Positivos Psilocibina_Mejora Introspección','Efectos Positivos Psilocibina_Mayor Satisfacción con la Vida','Efectos Positivos Psilocibina_Mejora Sueño','Efectos Positivos Psilocibina_Reducción Ansiedad','Efectos Positivos Psilocibina_Reducción Sintomas Depresión','Efectos Negativos Psilocibina_Sin Efecto Negativo']
efectos_moderados_psilocibina = ['Efectos Negativos Psilocibina_Cambios de humor','Efectos Negativos Psilocibina_Problemas Memoria o Atención']
efectos_negativos_determinantes_psilocibina = ['Efectos Negativos Psilocibina_Psicosis']



# Verifica si las columnas existen en el DF. Si no, devuelve un DF con valores predeterminados.
def get_columns(df, cols, default_value=False):
    existing_cols = [col for col in cols if col in df.columns]
    if not existing_cols:
        return pd.DataFrame({default_value: [default_value] * len(df)})
    return df[existing_cols]



# Conjuntos de Reglas para Nivel de Riesgo - Cannabis
def get_low_risk_cannabis(df_test):
    riesgo_bajo_cannabis = (
    (        
        # El consumo de cannabis no es frecuente.
            (
                (~df_test['Frecuencia Cannabis'].isin(['Diario', 'Varias veces por semana', 'Cada semana'])) &
                (
                    (
                        # No reporta dependencia a la sustancia ni consumo abusivo
                        (
                            (df_test['Dependencia Cannabis'] == False) &
                            (df_test['Abuso Cannabis'] == False)
                        ) &
                        # No presenta adicciones ni condiciones médicas riesgosas
                        (
                            get_columns(df_test, condiciones_medicas_adicciones).eq(False).all(axis=1) &
                            get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1)
                        )
                    ) &
                    # Ha experimentado efectos positivos con la sustancia y ningún efecto negativo determinante, como psicosis
                    get_columns(df_test, efectos_positivos_cannabis).eq(True).any(axis=1) &
                    get_columns(df_test, efectos_negativos_determinantes_cannabis).eq(False).all(axis=1) 
                    
                )
            ) |
            # El consumo de cannabis es frecuente pero ni el participante ni su familia cumplen con ninguna condición riesgosa ni moderada
            (
                (df_test['Frecuencia Cannabis'].isin(['Diario', 'Varias veces por semana', 'Cada semana'])) &
                (
                    # No reporta dependencia a la sustancia ni consumo abusivo
                    (
                        (df_test['Dependencia Cannabis'] == False) &
                        (df_test['Abuso Cannabis'] == False)
                    ) &
                    # No presenta adicciones ni condiciones médicas riesgosas
                    (
                        get_columns(df_test, condiciones_medicas_adicciones).eq(False).all(axis=1) &
                        get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1)
                    ) &
                    # Su familia no presenta adicciones ni condiciones médicas riesgosas
                    (
                        get_columns(df_test, historial_familiar_adicciones).eq(False).all(axis=1) &
                        get_columns(df_test, historial_familiar_condiciones_riesgosas).eq(False).all(axis=1)
                    )
                )
            )
        ) &
        # No presenta condiciones de riesgo
        (
            get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) &
            (df_test['Dependencia Cannabis'] == False) 
        )
    )
    
    return riesgo_bajo_cannabis

def get_medium_risk_cannabis(df_test):
    riesgo_medio_cannabis = (
    (            
        # El consumo no es muy frecuente
        (~df_test['Frecuencia Cannabis'].isin(['Diario', 'Varias veces por semana'])) &

            (     
                (
                        # No reporta dependencia a la sustancia ni consumo abusivo
                        (
                            (df_test['Dependencia Cannabis'] == False) |
                            (df_test['Abuso Cannabis'] == False)
                        ) &

                        # No presenta adicciones, ni condiciones medicas riesgosas 
                        (
                            get_columns(df_test, condiciones_medicas_adicciones).eq(False).all(axis=1) &
                            get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) 
                        ) |

                        # Alguien de su familia presenta una adicción o alguna condición riesgosa pero el participante no
                        (
                            (
                                get_columns(df_test, historial_familiar_condiciones_riesgosas).eq(True).any(axis=1) |
                                get_columns(df_test, historial_familiar_adicciones).eq(True).any(axis=1) 
                            ) &
                            (
                                get_columns(df_test, condiciones_medicas_adicciones).eq(False).all(axis=1) &
                                get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) 
                            ) 
                        )
                    ) &

                    # Experimenta efectos positivos con la sustancia y ningún efecto negativo determinante, como psicosis
                    get_columns(df_test, efectos_positivos_cannabis).eq(True).any(axis=1) &
                    get_columns(df_test, efectos_negativos_determinantes_cannabis).eq(False).all(axis=1) 
                    
                ) |

            # El consumo es frecuente pero no presenta condiciones riesgosas
            (df_test['Frecuencia Cannabis'].isin(['Diario', 'Varias veces por semana', 'Cada semana'])) &

            (
                # No reporta dependencia a la sustancia ni consumo abusivo
                (
                    (df_test['Dependencia Cannabis'] == False) &
                    (df_test['Abuso Cannabis'] == False)
                ) &

                # No presenta adicciones, ni condiciones medicas riesgosas 
                (
                    get_columns(df_test, condiciones_medicas_adicciones).eq(False).all(axis=1) &
                    get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) 
                ) |

                # Alguien de su familia presenta una adicción o alguna condición riesgosa pero el participante no
                (
                    (
                        get_columns(df_test, historial_familiar_condiciones_riesgosas).eq(True).any(axis=1) |
                        get_columns(df_test, historial_familiar_adicciones).eq(True).any(axis=1) 
                    ) &
                    (
                        get_columns(df_test, condiciones_medicas_adicciones).eq(False).all(axis=1) &
                        get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) 
                    ) 
                ) |

                # Reporta consumo abusivo o dependencia a la sustancia pero nunca ha experimentado efectos negativos determinantes ni tiene condiciones riesgosas
                (
                    (
                        (df_test['Dependencia Cannabis'] == True) |
                        (df_test['Abuso Cannabis'] == True)
                    ) & 
                    
                    get_columns(df_test, efectos_negativos_determinantes_cannabis).eq(False).all(axis=1) &
                    get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) 
                )

            )     
        ) &
        
        # No presenta condiciones de riesgo
        (
            get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) &
            (df_test['Dependencia Cannabis'] == False) 
        )
    
    )

    return riesgo_medio_cannabis

def get_high_risk_cannabis(df_test):
    riesgo_alto_cannabis = (
        # El consumo es frecuente y reporta condiciones riesgosas
        (    
            (df_test['Frecuencia Cannabis'].isin(['Diario', 'Varias veces a la semana', 'Cada semana'])) &

                (
                    # Reporta dependencia 
                    (
                        (df_test['Dependencia Cannabis'] == True) |

                        # Presenta efectos negativos moderados y adicciones o condiciones medicas riesgosas
                        (
                            get_columns(df_test, efectos_moderados_cannabis).eq(True).any(axis=1) &
                            get_columns(df_test, condiciones_medicas_adicciones).eq(True).any(axis=1) |
                            get_columns(df_test, condiciones_medicas_riesgosas).eq(True).any(axis=1) 
                        ) |

                        # Ha experimentado efectos negativos determinantes como psicosis y presenta condiciones riesgosas
                        (
                            get_columns(df_test, efectos_negativos_determinantes_cannabis).eq(True).any(axis=1) &
                            get_columns(df_test, condiciones_medicas_riesgosas).eq(True).any(axis=1) 
                        )
                    )
                )
        )  |

        # El consumo no es muy frecuente
        (    
            (~df_test['Frecuencia Cannabis'].isin(['Diario', 'Varias veces a la semana', 'Cada semana'])) &

                (
                    # Reporta dependencia o abuso
                    (
                        (df_test['Dependencia Cannabis'] == True) |
                        (df_test['Abuso Cannabis'] == True)
                    ) &

                        (
                            # Presenta adicciones y su familia presenta adicciones, condiciones riesgosas o condiciones moderadas
                            (   
                                get_columns(df_test, condiciones_medicas_adicciones).eq(True).any(axis=1) &
                                get_columns(df_test, historial_familiar_adicciones).eq(True).any(axis=1) |
                                get_columns(df_test, historial_familiar_condiciones_riesgosas).eq(True).any(axis=1) 
                            ) |

                            # Ha experimentado efectos negativos determinantes y presenta condiciones moderadas 
                            # Y su familia presenta adicciones, condiciones riesgosas o condiciones moderadas
                            (   
                                get_columns(df_test, efectos_negativos_determinantes_cannabis).eq(True).any(axis=1) &
                                get_columns(df_test, historial_familiar_adicciones).eq(True).any(axis=1) |
                                get_columns(df_test, historial_familiar_condiciones_riesgosas).eq(True).any(axis=1) 
                            )
                        )      
                ) |
                # Reporta una condición riesgosa o ha experimentado un efecto negativo determinante
                (
                    get_columns(df_test, condiciones_medicas_riesgosas).eq(True).any(axis=1)  |
                    get_columns(df_test, efectos_negativos_determinantes_cannabis).eq(True).any(axis=1)  
                )
        ) 
    )  

    return riesgo_alto_cannabis  


# Conjuntos de Reglas para Nivel de Riesgo - Cannabis
def get_low_risk_psilocibina(df_test):
    riesgo_bajo_psilocibina = (
        (    
            # Ha consumido psilocibina en macrodosis
            (    
                (df_test['Tipo de Dosis'] == 'Macrodosis') &

                    (    
                        (     
                            # Ha realizado dos o más tratamiento 
                            (
                                (df_test['Cantidad Tratamientos'].isin(['Dos', 'Más de tres'])) &
                                    (
                                        # La calificación dada al tratamiento es de 4 o 5
                                        (df_test['Calificación Tratamiento'].isin([4,5]))
                                    )
                            ) |

                            (
                                # Ha consumido psilocibina con fines terapeuticos
                                (df_test['Propósito Psilocibina'].isin(['Fines terapéuticos', 'Ambos']))
                            ) 

                        ) &

                        (    
                            # Ni el participante ni su familia presentan adicciones o condiciones medicas riesgosas
                            (
                                get_columns(df_test, condiciones_medicas_adicciones).eq(False).all(axis=1)  &
                                get_columns(df_test, historial_familiar_adicciones).eq(False).all(axis=1)  &
                                get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1)  &
                                get_columns(df_test, historial_familiar_condiciones_riesgosas).eq(False).all(axis=1)  
                            )  &
                            # No reporta dependencia a la sustancia ni consumo abusivo
                            (
                                (df_test['Dependencia Psilocibina'] == False) &
                                (df_test['Abuso Psilocibina'] == False) 
                            ) &
                            # Ha experimentado efectos positivos con la sustancia y ningún efecto negativo determinante, como psicosis
                            (
                                get_columns(df_test, efectos_negativos_determinantes_psilocibina).eq(False).all(axis=1)  &
                                get_columns(df_test, efectos_positivos_psilocibina).eq(True).any(axis=1)  
                            )

                        )
                    ) 
            ) |

            # No ha consumido en macrodosis pero cumple con las condiciones sanas
            # Ni el participante ni su familia presentan condiciones riesgosas 
            (
                get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1)  &
                get_columns(df_test, historial_familiar_condiciones_riesgosas).eq(False).all(axis=1)  
            )  &
            # No reporta dependencia a la sustancia ni consumo abusivo
            (
                (df_test['Dependencia Psilocibina'] == False) &
                (df_test['Abuso Psilocibina'] == False) 
            ) &
            # Ha experimentado efectos positivos con la sustancia y ningún efecto negativo determinante, como psicosis
            (
                get_columns(df_test, efectos_negativos_determinantes_psilocibina).eq(False).all(axis=1)  &
                get_columns(df_test, efectos_positivos_psilocibina).eq(True).any(axis=1)  
            )
        ) &

        # No presenta condiciones de riesgo
        (
            get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) &
            (df_test['Dependencia Psilocibina'] == False) 
        )
    )

    return riesgo_bajo_psilocibina

def get_medium_risk_psilocibina(df_test):

    riesgo_medio_psilocibina = (
        
        (    
            # Ha consumido psilocibina en macrodosis
            (    
                (df_test['Tipo de Dosis'] == 'Macrodosis') &

                    (     
                        # Ha realizado tratamientos
                        (
                            (df_test['Cantidad Tratamientos'] != 'Sin Dato') &
                                (
                                    # La calificación dada al tratamiento es diferente a 1
                                    (df_test['Calificación Tratamiento'] != 1)
                                )
                        ) |

                        (
                            # Ha consumido psilocibina con fines terapeuticos
                            (df_test['Propósito Psilocibina'].isin(['Fines terapéuticos', 'Ambos']))
                        ) 

                    ) |

                    (    
                        # No presenta condiciones riesgosas 
                        (
                            get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1)  
                        )  &
                        # no es dependiente ni abusa de la sustancia
                        (
                            (df_test['Dependencia Psilocibina'] == False) &
                            (df_test['Abuso Psilocibina'] == False) 
                        ) &
                        # No reporta dependencia a la sustancia ni consumo abusivo
                        (
                            get_columns(df_test, efectos_negativos_determinantes_psilocibina).eq(False).all(axis=1)  &
                            get_columns(df_test, efectos_positivos_psilocibina).eq(True).any(axis=1)  
                        )

                    )

            ) |

            # No ha consumido en macrodosis pero cumple con las condiciones sanas
            # No presenta condiciones riesgosas 
            (
                get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1)  
            )  &
            # No reporta dependencia a la sustancia ni consumo abusivo
            (
                (df_test['Dependencia Psilocibina'] == False) &
                (df_test['Abuso Psilocibina'] == False) 
            ) &
            # Ha experimentado efectos positivos con la sustancia y ningún efecto negativo determinante, como psicosis
            (
                get_columns(df_test, efectos_negativos_determinantes_psilocibina).eq(False).all(axis=1)  &
                get_columns(df_test, efectos_positivos_psilocibina).eq(True).any(axis=1)  
            )
        ) &

        # No presenta condiciones de riesgo
        (
            get_columns(df_test, condiciones_medicas_riesgosas).eq(False).all(axis=1) &
            (df_test['Dependencia Psilocibina'] == False) 
        )
    )

    return riesgo_medio_psilocibina

def get_high_risk_psilocibina(df_test):
    riesgo_alto_psilocibina = (
    
        # Ha consumido psilocibina en macrodosis
        (           
            (df_test['Tipo de Dosis'] == 'Macrodosis') &

                (     
                    # Ha realizado tratamientos y ha dado una mala calificación
                    (
                        (df_test['Cantidad Tratamientos'] != 'Sin Dato') &
                        (df_test['Calificación Tratamiento'] == 1)
                    )
                )  &

                (    
                    # El participante y su familia presentan cualquier condición riesgosa
                    (
                        get_columns(df_test, condiciones_medicas_riesgosas).eq(True).any(axis=1)  &
                        get_columns(df_test, historial_familiar_condiciones_riesgosas).eq(True).any(axis=1)  
                    )  |
                    # Reporta dependencia a la sustancia
                    (
                        (df_test['Dependencia Psilocibina'] == True)  
                    ) | 
                    # Ha experimentado efectos negativos determinantes
                    (
                        get_columns(df_test, efectos_negativos_determinantes_psilocibina).eq(True).any(axis=1)  
                    )

                )
                    
        ) |

        # No ha consumido en macrodosis pero no cumple con las condiciones sanas
        (
            # Presenta cualquier condición riesgosa
            (
                get_columns(df_test, condiciones_medicas_riesgosas).eq(True).any(axis=1)  
            )  |
            # Reporta dependencia a la sustancia
            (
                (df_test['Dependencia Psilocibina'] == False) 
            ) |
            # Ha experimentado efectos negativos determinantes
            (
                get_columns(df_test, efectos_negativos_determinantes_psilocibina).eq(True).any(axis=1)  
            )
        ) 
    )

    return riesgo_alto_psilocibina