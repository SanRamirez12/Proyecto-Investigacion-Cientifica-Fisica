#Librerias de utils, y otras importantes:
import utils_model_dev as umd
import time
import numpy as np
from livelossplot import PlotLossesKeras

#Metodos de Skelearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

#Metodos de Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping



#Empezamos leyendo los archivos. tomamos el datos sin fuentes no asociadas:
X, Y, encoder = umd.cargar_dataset('df_final_sin_UncAss.parquet', encoding='label', return_encoder=True)
# print(f'Features: {X} y target Values: {Y}')

#Se aseguran que son tipos correctos
X = X.copy()
Y = Y.astype(np.int32)
#print(f'Valores de X: \n{X}\n, Y: \n{Y}')

#Separamos el test set (15%) del training/validations sets (85%)
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=47)
print(f'\n Training/Validation sets shapes: X:{X_trainval.shape} y Y:{Y_trainval.shape} \n Testing sets shapes: X:{X_test.shape} y Y:{Y_test.shape}')

#Se identifican las columnas a no escalar (spectrum types) y las demas se escalan 
cols_no_escaladas = [col for col in X.columns if col.startswith("spectrum_")]
cols_a_escalar = [col for col in X.columns if col not in cols_no_escaladas]

#Se define el K-Fold para el cross validation, en este caso de estratificado para el imbalance de clases
kfold = StratifiedKFold(
    n_splits=5, #Numero de folds
    shuffle=True, #Baraja los datos antes de dividirlos
    random_state=47 #Semilla de generador de numero aleatorio
    )


t_train_inicio = time.time() #Se define un tiempo inicial para entrenamiento

#Listas importantes para llevar registros de informacion por folds:
histories = [] #Lista para guardar las historias de entrenamiento por fold
fold_accuracies = [] #Lista de exactitudes por fold
fold_classification_reports = [] #Lista de los reportes de clasificacion por fold
pesos_por_fold = [] #Lista de los pesos de cada clase por fold
modelos_entrenados = [] #Lista de modelos entrenados por cada fold
scalers_folds = [] #Guardamos los escaladores de cada fold 

#################################### INICIO DE LOOP ##########################################
##############################################################################################

#Se realiza el Loop de Cross-Validation:
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_trainval, Y_trainval)):
    print(f"\nFold #{fold_idx + 1}\n")
    
    #Se define el entrenamiento de cada valor por indice de training y cv sets
    # Dividimos los índices
    X_train_fold = X_trainval.iloc[train_idx].copy()
    Y_train_fold = Y_trainval[train_idx]
    X_val_fold = X_trainval.iloc[val_idx].copy()
    Y_val_fold = Y_trainval[val_idx]

    #Se define el scaler a usar:
    scaler = StandardScaler()
    
    #Se copian los datasets de training y validation para dicho fold:
    X_train_fold_scaled = X_train_fold.copy()
    X_val_fold_scaled = X_val_fold.copy()
    
    #Se escalan los datos:
    X_train_fold_scaled[cols_a_escalar] = scaler.fit_transform(X_train_fold[cols_a_escalar])
    X_val_fold_scaled[cols_a_escalar] = scaler.transform(X_val_fold[cols_a_escalar])
    
    #Se verifica que sean del tipo deseado
    X_train_final = X_train_fold_scaled.values.astype(np.float32)
    X_val_final = X_val_fold_scaled.values.astype(np.float32)

    #Se guarda el scaler
    scalers_folds.append(scaler)
    
    #Se realiza un calculo de los pesos de clase para este fold
    #Factores personalizados para cada clase 
    # factores_pesos = {
    #     0: 13.365080916417318,
    #     1: 9.160866709538478,
    #     2: 6.015617622454988,
    #     3: 13.921084016240723,
    #     4: 1.2428873086003254
    # }
    
    #Cálculo de los pesos balanceados
    clases_unicas = np.unique(Y_train_fold)
    pesos_balanceados = compute_class_weight(class_weight='balanced',
                                             classes=clases_unicas,
                                             y=Y_train_fold)
    
    #Multiplicación por los factores personalizados
    diccionario_de_pesos = {
        clase: peso_balanceado #* factores_pesos[clase]
        for clase, peso_balanceado in zip(clases_unicas, pesos_balanceados)
    }
    
    pesos_por_fold.append(diccionario_de_pesos) #Guardar pesos en lista por fold
    
    #Hiperparametros de arquitectura
    hidden_units = [93, 97, 90]
    hidden_act_funct = ['tanh', 'relu', 'elu']
    tasa_abandono = [0.05123092402361263, 0.342663498113033, 0.22918274282264045]
    
    #Se crea la arquitectura del modelo
    model = Sequential([
        Dense(hidden_units[0], input_shape=(X.shape[1],), activation=hidden_act_funct[0], name='hidden_layer1'),
        Dropout(tasa_abandono[0]),
        Dense(hidden_units[1], activation=hidden_act_funct[1], name='hidden_layer2'),
        Dropout(tasa_abandono[1]),
        Dense(hidden_units[2], activation=hidden_act_funct[2], name='hidden_layer3'),
        Dropout(tasa_abandono[2]),
        Dense(5, activation='softmax', name='output_layer')
    ])
    
    #Se compila el modelo
    tasa_aprendizaje = 0.0010986019625951863
    optimizador = RMSprop(learning_rate=tasa_aprendizaje)
    
    model.compile(optimizer=optimizador,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    #Resumen del modelo
    model.summary()
    
    #Se hace un Callback de con parada anticipada en caso de que el error de validacion no mejore
    parada_temprana = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    ####################### Entrenamiento ####################################
    
    #Se entrena el modelo:
    print('\nEntrenando .......\n')
    
    #Se establece un tiempo inicial de entrenamiento, para este fold
    t0 = time.time()
    
    history = model.fit(X_train_final, Y_train_fold,
              validation_data=(X_val_final, Y_val_fold),
              epochs=1000,  #Epocas de entrenamiento
              batch_size=49, #Tamano del lote 
              callbacks=[PlotLossesKeras(), parada_temprana], #Metemos la parada temprana y grafico a tiempo real del Learning curve por fold
              class_weight=diccionario_de_pesos, #Pesos para las clases
              verbose=1)
    histories.append(history.history) #Se agraga la historia de entrenamiento a las historias totales
    modelos_entrenados.append(model) #Guardamos el modelo en la lista de modelos entrenados
    
    tf = time.time()
    minutos_xfold = int((tf- t0)// 60)
    segundos_xfold = int((tf- t0) % 60)
    print(f"\nDuración entrenamiento en el Fold #{fold_idx+1}:\n {minutos_xfold} minutos y {segundos_xfold} segundos\n")
    
    ####################### Evaluacion con cv set#######################################
    
    #Se realiza la evaluación en validación (y luego se puede guardar métricas por fold)
    val_loss, val_acc = model.evaluate(X_val_final, Y_val_fold, verbose=0)
    fold_accuracies.append(val_acc) #Se agrega la exactitud a la lista de exactitudes
    print(f"\nExactitud de la validación en el Fold #{fold_idx+1}: {val_acc:.4f}\n")
    
    y_pred_probs = model.predict(X_val_final, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    #Se printea el reporte de classificacion de sklearn para ver las metricas y como funciona el modelo por fold
    print(f"\n Reporte de clasificación - Fold {fold_idx+1}\n")
    report = classification_report(Y_val_fold, y_pred, target_names=encoder.classes_, output_dict=True)
    fold_classification_reports.append(report) #Se agrega el reporte de clasificacion a la lista respectiva
    print(classification_report(Y_val_fold, y_pred, target_names=encoder.classes_))
    
    #Printeamos la matriz de confusion:
    umd.matriz_confusion(Y_val_fold, y_pred, encoder.classes_)

#################################### FIN DE LOOP #############################################
##############################################################################################


t_train_fin = time.time() #Tiempo final de entrenamiento
duracion = t_train_fin - t_train_inicio
minutos = int(duracion // 60)
segundos = int(duracion % 60)
print(f"\nDuración entrenamiento: {minutos} minutos y {segundos} segundos\n")


####################### Evaluacion con test set ###################################################################

# Evaluación final sobre el test set con ensemble
print("\n\n########## EVALUACIÓN FINAL SOBRE TEST SET ##########")

#Promedio de probabilidades de predicción sobre X_test
X_test_pred_probs_folds = []

#Realiza un ensemble de predicciones
for i in range(len(modelos_entrenados)):
    #Primero se transforma un test set apartir del scaler del modelo de un fold especifico y su scaler respectivo
    scaler = scalers_folds[i]
    X_test_scaled = X_test.copy()
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test_scaled[cols_a_escalar])
    X_test_final = X_test_scaled.values.astype(np.float32)
    
    #Se predice y se agrega ala lista de predicciones sobre el test set
    y_test_probs = modelos_entrenados[i].predict(X_test_final, verbose=0)
    X_test_pred_probs_folds.append(y_test_probs)

#Se tiene un ensemble de predicciones: promedio de softmax
y_test_mean_probs = np.mean(X_test_pred_probs_folds, axis=0)
y_test_pred = np.argmax(y_test_mean_probs, axis=1)

#Se da una Evaluación final con sus respectivas metricas de todo el proceso de entrenamiento.
print("\nReporte de Clasificación - Ensemble en Test Set")
print(classification_report(Y_test, y_test_pred, target_names=encoder.classes_))
umd.matriz_confusion(Y_test, y_test_pred, encoder.classes_)


#################################### METRICAS DE EVALUACION GENERALES #############################################
#Printeamos las learning curves
umd.learning_curves(histories)

#Printeamos los pesos de las clases por fold (hiperparamatros mas relevantes despues de lr):
umd.mostrar_pesos_fold(pesos_por_fold, encoder)
umd.graficar_pesos_por_fold(pesos_por_fold, encoder)    

#Printeamos un reporte general de metricas:
umd.reporte_general_metricas(fold_accuracies, fold_classification_reports, encoder)

#Ploteo de analisis de varianza de los folds mediante los reportes de clasificacion:
umd.varianza_entre_folds(fold_classification_reports)