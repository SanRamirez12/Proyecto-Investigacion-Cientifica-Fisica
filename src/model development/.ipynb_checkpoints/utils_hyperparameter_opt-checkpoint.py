import os
from tensorflow.keras import optimizers
import joblib
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Funcion para obtener optimizador:
#Usamos como parametros del optimizador su nombre, el learning rate y el momentum en caso de que el optimizador lo ocupe
def get_optimizer(name, lr, momentum):
    return {
        'SGD': optimizers.SGD(learning_rate=lr),
        'SGD_momentum': optimizers.SGD(learning_rate=lr, momentum=momentum),
        'SGD_NAG': optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True),
        'RMSprop': optimizers.RMSprop(learning_rate=lr),
        'Adagrad': optimizers.Adagrad(learning_rate=lr),
        'Adadelta': optimizers.Adadelta(learning_rate=lr),
        'Adam': optimizers.Adam(learning_rate=lr),
        'AdamW': optimizers.AdamW(learning_rate=lr),
        'Nadam': optimizers.Nadam(learning_rate=lr)
    }[name]


def guardar_estudio_optuna(study, nombre_archivo):
   # Obtener ruta del directorio actual del script
    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    # Construir ruta hacia data/hyperparameter studies
    carpeta_salida = os.path.join(directorio_actual, '..', '..', 'data', 'hyperparameter studies')
    carpeta_salida = os.path.abspath(carpeta_salida)  # Normaliza la ruta

    # Crear la carpeta si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Ruta final del archivo
    ruta_archivo = os.path.join(carpeta_salida, f"{nombre_archivo}.pkl")

    # Guardar el estudio
    joblib.dump(study, ruta_archivo)
    print(f"Estudio exportado correctamente a: {ruta_archivo}")

#Muestra cómo fue evolucionando la métrica objetivo (F1-score) a lo largo de los trials.    
def graf_registros_optimizacion(study):
    f1_scores = [t.value for t in study.trials if t.value is not None]

    plt.figure(figsize=(10, 5))
    plt.plot(f1_scores, marker='o', linestyle='-')
    plt.title("Evolución del F1-score (weighted) por trial")
    plt.xlabel("Trial")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
 
#Muestra qué hiperparámetros tuvieron mayor impacto en la métrica objetivo.
def graf_importancia_hyperparametros(study):
    importances = optuna.importance.get_param_importances(study)
    df_importances = pd.DataFrame(list(importances.items()), columns=["Parametro", "Importancia"])

    plt.figure(figsize=(10, 5))
    plt.barh(df_importances["Parametro"], df_importances["Importancia"], color='steelblue')
    plt.title("Importancia estimada de los hiperparámetros")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.grid(True, axis='x')
    plt.show()

#Metodo para cargar studies (va a ser utilizado en los notebooks)
def cargar_study_desde_pkl(ruta_archivo):

    try:
        study = joblib.load(ruta_archivo)
        print(f"Estudio cargado correctamente desde: {ruta_archivo}")
        return study
    except Exception as e:
        print(f"Error al cargar el estudio: {e}")
        return None