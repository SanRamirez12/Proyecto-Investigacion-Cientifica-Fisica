'''
    Optimizacion de hiperparametros con Optuna
'''
#Librerias de utils, y otras importantes:
import utils_model_dev as umd
import utils_hyperparameter_opt as uho
import time
import numpy as np
import optuna

#Metodos de Skelearn
from sklearn.model_selection import train_test_split    
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

#Metodos Tensorflow
from tensorflow import keras
from tensorflow.keras import layers

#Metodos Optuna:
from optuna.integration import TFKerasPruningCallback


# ==================== CARGA Y ESCALADO ====================

#Empezamos leyendo los archivos. tomamos el datos sin fuentes no asociadas:
X, Y, encoder = umd.cargar_dataset('df_final_sin_UncAss.parquet', encoding='label', return_encoder=True)

#Se aseguran que son tipos correctos
X = X.copy()
Y = Y.astype(np.int32)

#Separamos el test set (20%) del training/validations sets (80%)
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

#Se identifican las columnas a no escalar (spectrum types) y las demas se escalan 
cols_no_escaladas = [col for col in X.columns if col.startswith("spectrum_")]
cols_a_escalar = [col for col in X.columns if col not in cols_no_escaladas]

#Se escalan los datasets
X_temp_final, X_test_final = umd.escalar_datasets(X_temp, X_test, cols_a_escalar)

# ==================== SUBMUESTREO PARA OPTIMIZACIÓN ====================
X_sub, _, Y_sub, _ = train_test_split(X_temp_final, Y_temp, train_size=0.5, stratify=Y_temp, random_state=42)


# ==================== OPTUNA OBJECTIVE ====================
#Se define la funcion objetivo crucial para los porcesos de optimizacion con optuna
def objective(trial):
    
    # Hiperparámetros de numero decapas
    num_layers = trial.suggest_int('num_capas_ocultas', 1, 3) 
    
    #Hiperparametros de compilacion
    optimizer_name = trial.suggest_categorical('optimizador', [
        'SGD', 'SGD_momentum', 'SGD_NAG', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'AdamW', 'Nadam'
    ])
    learning_rate = trial.suggest_float('tasa_aprendizaje', 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 0.9) if 'momentum' in optimizer_name else 0.0
    batch_size = trial.suggest_int('tamaño_lote', 32, 128)
    
    # Pesos por clase como hiperparámetros nombrados según el nombre real de la clase
    base_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_sub), y=Y_sub)
    
    # Genera los factores de ponderación como hiperparámetros nombrados con los nombres de clase reales
    class_weight_factors = {}
    for i, class_name in enumerate(encoder.classes_):
        param_name = f'peso_de_{class_name}'
        class_weight_factors[i] = trial.suggest_float(param_name, 0.1, 15.0)
    
    #Se multiplican los factores por los pesos base
    class_weights = {i: base_weights[i] * class_weight_factors[i] for i in range(len(encoder.classes_))}
    

    #Se plantea el modelo de ANN tipo MLP:
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_sub.shape[1],)))
    
    for i in range(num_layers):
        units = trial.suggest_int(f'unidades_capa_{i}', 30, 130)
        activation = trial.suggest_categorical(f'activación_capa_{i}', ['relu', 'tanh', 'selu', 'elu', 'gelu'])
        dropout_rate = trial.suggest_float(f'tasa_abandono_capa_{i}', 0.0, 0.5)
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout_rate))

    #Se agrega el output layer con activacion softmax:
    model.add(layers.Dense(5, activation='softmax'))
    
    #Se compila el modelo
    model.compile(
        optimizer=uho.get_optimizer(optimizer_name, learning_rate, momentum),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #Se entena el modelo para un trial especifico con hiperparametros especificos
    t_inicio = time.time() #tiempo inical por dicho trial
    history = model.fit(
        X_sub, Y_sub,
        validation_split=0.2,
        epochs=200,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            TFKerasPruningCallback(trial, 'val_accuracy')
        ],
        class_weight=class_weights
    )
    t_final = time.time()
    duracion = t_final- t_inicio
    minutos = int(duracion // 60)
    segundos = int(duracion % 60)
    print(f"\Duración del trial:\n {minutos} minutos y {segundos} segundos\n")
    #Proceso de evaluación interna para ese modelo de ese trial:
    y_pred = model.predict(X_sub)
    y_pred_labels = np.argmax(y_pred, axis=1)
    #Muestra el f1 score apartir de 
    f1 = f1_score(Y_sub, y_pred_labels, average='weighted')
    return f1

# ==================== EJECUCIÓN DEL ESTUDIO ====================
#Parte donde se buscan los mejores modelos con los mejores hiperparametros
t0 = time.time()
#Se crea el estudio con optimizacion bayesiana de Optuna
study = optuna.create_study(
    direction='maximize', #Maximiza la funcion objetivo para encontrar el modelo con el mejor f1score
    study_name='AGN_opt_inicial_prueba', #Nombre del estudio
    sampler=optuna.samplers.TPESampler(), #Tree-structured Parzen Estimator como estrategia de muestreo
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)) #Permite interrumpir los trials que no prometen buenos resultados,
    #MedianPruner compara la métrica de validación con la mediana de otros trials en ese punto, 
    #n_warmup_steps=5: los primeros 5 pasos (épocas) de cada trial no se interrumpen

#Se ejecuta el estudio
study.optimize(objective, n_trials=200)

#Tiempos de estudio:
tf = time.time()
duracion = tf- t0
minutos = int(duracion // 60)
segundos = int(duracion % 60)
print(f"\nDuración de ejecución del estudio:\n {minutos} minutos y {segundos} segundos\n")

# ==================== VISUALIZACIÓN Y GUARDADO ====================
#Se guarda el estudio en la carpeta de hyperparameter studies:
nombre_studie_salida = "optuna_agn_study_200trials_pesos_nombrados"
uho.guardar_estudio_optuna(study, nombre_studie_salida)

#Describimos el mejor trial
print("\nMejor trial:")
trial = study.best_trial #Se extrae el mejor trial del estudi
print(f"  Valor (F1 pesado): {trial.value}")
print("  Hiperparametros:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


#Visualizaciones de Optuna:
uho.graf_registros_optimizacion(study)
uho.graf_importancia_hyperparametros(study)

