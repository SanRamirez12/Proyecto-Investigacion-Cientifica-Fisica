#Librerias importantes
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
from tensorflow.keras import optimizers

#Metodos y Libreria de Optuna
import optuna
import optuna.visualization as vis


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
def cargar_study_desde_ruta(ruta_archivo):
    print(f"Intentando cargar archivo desde: {ruta_archivo}")

    if not os.path.exists(ruta_archivo):
        print("Archivo no encontrado. Verifica la ruta proporcionada.")
        return None

    try:
        study = joblib.load(ruta_archivo)
        print("Estudio cargado correctamente.")
        return study
    except Exception as e:
        print(f"Error al cargar el estudio: {e}")
        return None
    
# Exportar los mejores N trials a CSV
def exportar_top_trials_a_csv(study, top_n=10):
    #Crea la ruta de destino para los archivos:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    carpeta_salida = os.path.join(directorio_actual, '..', '..', 'data', 'hyperparameter studies')
    carpeta_salida = os.path.abspath(carpeta_salida)
    os.makedirs(carpeta_salida, exist_ok=True)

    #Busca cuales son los mejores trials
    top_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)[:top_n]
    registros = []
    for i, trial in enumerate(top_trials):
        fila = {'Trial #': trial.number, 'F1 Score': trial.value}
        fila.update(trial.params)
        registros.append(fila)

    #Los convierte en un DF y luego a un csv
    df = pd.DataFrame(registros)
    ruta_csv = os.path.join(carpeta_salida, f"top_{top_n}_trials_{study.study_name}.csv")
    df.to_csv(ruta_csv, index=False)
    print(f"Top {top_n} trials exportados a: {ruta_csv}")
    
#Metodo que genera visualizaciones de optuna en html, abriendo en el navegador
def generar_visualizaciones_optuna(study, nombre_custom=""):
    #Se abre en el browser por default ya que spyder no cuenta con plots interactivos
    pio.renderers.default = 'browser'

    #Se guardan en carpeta optuna visualizers
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    carpeta_salida = os.path.join(directorio_actual, '..', '..', 'optuna visualizers')
    os.makedirs(carpeta_salida, exist_ok=True)

    # Diccionario con los gráficos que querés exportar
    graficos = {
        f"optuna_slice_{nombre_custom}.html": vis.plot_slice(study),
        f"optuna_optimization_history_{nombre_custom}.html": vis.plot_optimization_history(study),
        f"optuna_param_importances_{nombre_custom}.html": vis.plot_param_importances(study),
        f"optuna_parallel_coordinate_{nombre_custom}.html": vis.plot_parallel_coordinate(study)
    }

    for nombre_archivo, fig in graficos.items():
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
        fig.show()
        fig.write_html(ruta_salida)

    print(f"Visualizaciones de Optuna guardadas en: {carpeta_salida}")    
    
    
    
    
    
    
    
    
