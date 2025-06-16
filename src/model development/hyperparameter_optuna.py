'''
    Optimizacion de hiperparametros con Optuna
'''
#Librerias de utils, y otras importantes:
import utils_model_dev as umd
import utils_hyperparameter_opt as uho
import time
import numpy as np

#Metodos de Skelearn
from sklearn.model_selection import train_test_split    
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

#Metodos Tensorflow
from tensorflow import keras
from tensorflow.keras import layers

#Metodos Optuna:
import optuna
from optuna.integration import TFKerasPruningCallback

# ==================== CARGA Y LECTURA ====================

#Empezamos leyendo los archivos. tomamos el datos sin fuentes no asociadas:
X, Y, encoder = umd.cargar_dataset('df_final_solo_clases_definidas.parquet', encoding='label', return_encoder=True)

#Se aseguran que son tipos correctos
X = X.copy()
Y = Y.astype(np.int32)
n_clases = len(encoder.classes_) #Cantidad de clases

#Se identifican las columnas a no escalar (spectrum types) y las demas se escalan 
cols_no_escaladas = [col for col in X.columns if col.startswith("spectrum_")]
cols_a_escalar = [col for col in X.columns if col not in cols_no_escaladas]


# ==================== OPTUNA OBJECTIVE ====================
#Se define la funcion objetivo crucial para los porcesos de optimizacion con optuna
def objective(trial):
    
    # Hiperparámetros de numero decapas
    num_layers = trial.suggest_int('num_capas_ocultas', 3, 5) 
    
    #Hiperparametros de compilacion
    optimizer_name = trial.suggest_categorical('optimizador', [
        'SGD', 'SGD_momentum', 'SGD_NAG', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'AdamW', 'Nadam'
    ])
    learning_rate = trial.suggest_float('tasa_aprendizaje', 1e-5, 1e-3, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 0.9, step=0.05) if 'momentum' in optimizer_name else 0.0
    batch_size = trial.suggest_int('tamaño_lote', 32, 128)
    
    #Pesos por clase como hiperparámetros nombrados según el nombre real de la clase
    class_weights = {}
    for i, clase in enumerate(encoder.classes_):
        if clase == 'BLL':
            class_weights[i] = trial.suggest_float('peso_BLL', 0.5, 5.0)
        elif clase == 'FSRQ':
            class_weights[i] = trial.suggest_float('peso_FSRQ', 0.5, 5.0)
        elif clase == 'NoAGN':
            class_weights[i] = trial.suggest_float('peso_NoAGN', 0.5, 5.0)
        elif clase == 'OtroAGN':
            class_weights[i] = trial.suggest_float('peso_OtroAGN', 0.5, 25.0)
    
    # ===== Split train/val manual (80/20) =====
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    
    # ===== Escalado (fit solo en train, transform en ambos) =====
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_val_scaled[cols_a_escalar] = scaler.transform(X_val[cols_a_escalar])

    #Se plantea el modelo de ANN tipo MLP:
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1],)))
    for i in range(num_layers):
        units = trial.suggest_int(f'unidades_capa_{i}', 30, 150)
        activation = trial.suggest_categorical(f'activación_capa_{i}', ['relu', 'tanh', 'selu', 'elu', 'gelu'])
        dropout_rate = trial.suggest_float(f'tasa_abandono_capa_{i}', 0.0, 0.5, step=0.05)
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout_rate))

    #Se agrega el output layer con activacion softmax:
    model.add(layers.Dense(n_clases, activation='softmax'))
    
    #Se compila el modelo
    model.compile(
        optimizer=uho.get_optimizer(optimizer_name, learning_rate, momentum),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ===== Entrenamiento del modelo para un trial especifico con hiperparametros especificos=====
    t_inicio = time.time() #tiempo inical por dicho trial
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=500,
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
    print(f"\Duración del trial:\n {int(duracion // 60)} minutos y {int(duracion % 60)} segundos\n")
    
    # ===== Evaluación final sobre validation ara ese modelo de ese trial ===== 
    y_pred = model.predict(X_val_scaled)
    y_pred_labels = np.argmax(y_pred, axis=1)
    #Muestra el f1 score apartir de 
    f1 = f1_score(y_val, y_pred_labels, average='weighted') #Puede ser: 'weighted', 'macro' y  'micro'
    return f1

# ==================== EJECUCIÓN DEL ESTUDIO ====================
#Parte donde se buscan los mejores modelos con los mejores hiperparametros
t0 = time.time()
#Se crea el estudio con optimizacion bayesiana de Optuna
study = optuna.create_study(
    direction='maximize', #Maximiza la funcion objetivo para encontrar el modelo con el mejor f1score
    study_name='Study600trials_#2_F1weighted_suggWeights', #Nombre del estudio
    sampler=optuna.samplers.TPESampler(), #Tree-structured Parzen Estimator como estrategia de muestreo
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)) #Permite interrumpir los trials que no prometen buenos resultados,
    #MedianPruner compara la métrica de validación con la mediana de otros trials en ese punto, 
    #n_warmup_steps=5: los primeros 5 pasos (épocas) de cada trial no se interrumpen

#Se ejecuta el estudio
study.optimize(objective, n_trials=600, timeout=5400) #Time out de 1.5h

#Tiempos de estudio:
tf = time.time()
duracion = tf- t0
print(f"\nDuración de ejecución del estudio:\n {int(duracion // 60)} minutos y {int(duracion % 60)} segundos\n")

# ==================== VISUALIZACIÓN Y GUARDADO ====================
#Se guarda el estudio en la carpeta de hyperparameter studies:
nombre_studie_salida = "Study600trials_#2_F1weighted_suggWeights"
uho.guardar_estudio_optuna(study, nombre_studie_salida)
uho.exportar_top_trials_a_csv(study, 25) 

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
uho.generar_visualizaciones_optuna(study, '600trials')