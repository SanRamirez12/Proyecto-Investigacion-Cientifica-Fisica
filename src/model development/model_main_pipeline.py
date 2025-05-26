#Librerias de utils, y otras importantes:
import utils_model_dev as umd
import time
import numpy as np
from livelossplot import PlotLossesKeras

#Metodos de Skelearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

#Metodos de Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



#Empezamos leyendo los archivos. tomamos el datos sin fuentes no asociadas:
X, Y, encoder = umd.cargar_dataset('df_final_sin_UncAss.parquet', encoding='label', return_encoder=True)
# print(f'Features: {X} y target Values: {Y}')

#Se aseguran que son tipos correctos
X = X.copy()
Y = Y.astype(np.int32)
#print(f'Valores de X: \n{X}\n, Y: \n{Y}')

#Separamos el test set (20%) del training/validations sets (80%)
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
print(f'\n Training/Validation sets shapes: X:{X_temp.shape} y Y:{Y_temp.shape} \n Testing sets shapes: X:{X_test.shape} y Y:{Y_test.shape}')

#Se identifican las columnas a no escalar (spectrum types) y las demas se escalan 
cols_no_escaladas = [col for col in X.columns if col.startswith("spectrum_")]
cols_a_escalar = [col for col in X.columns if col not in cols_no_escaladas]

#Se escalan los datasets
X_temp_final, X_test_final = umd.escalar_datasets(X_temp, X_test, cols_a_escalar)


#Se define el K-Fold para el cross validation, en este caso de estratificado para el imbalance de clases
kfold = StratifiedKFold(
    n_splits=5, #Numero de folds
    shuffle=True, #Baraja los datos antes de dividirlos
    random_state=42 #Semilla de generador de numero aleatorio
    )


t_train_inicio = time.time() #Se define un tiempo inicial para entrenamiento

histories = [] #Lista para guardar las historias de entrenamiento por fold
fold_accuracies = [] #Lista de exactitudes por fold
fold_classification_reports = [] #Lista de los reportes de clasificacion por fold
#Se realiza el Loop de Cross-Validation:
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_temp_final, Y_temp)):
    print(f"\nFold #{fold_idx + 1}\n")
    
    #Se define el entrenamiento de cada valor por indice de training y cv sets
    X_train_fold = X_temp_final[train_idx]
    Y_train_fold = Y_temp[train_idx]
    X_val_fold = X_temp_final[val_idx]
    Y_val_fold = Y_temp[val_idx]

    #Se realiza un calculo de los pesos de clase para este fold
    pesos_de_clase = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(Y_train_fold),
                                         y=Y_train_fold)
    diccionario_de_pesos = dict(zip(np.unique(Y_train_fold), pesos_de_clase))

    #Se crea la arquitectura del modelo
    model = Sequential([
        Dense(50, input_shape=(X.shape[1],), activation='relu', name='hidden_layer_1'),
        #Dense(32, activation='relu', name='hidden_layer_2'),
        Dense(5, activation='softmax', name='output_layer')
    ])
    
    #Se compila el modelo
    model.compile(optimizer=Adam(learning_rate=0.001),
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
    
    history = model.fit(X_train_fold, Y_train_fold,
              validation_data=(X_val_fold, Y_val_fold),
              epochs=1000,  #Epocas de entrenamiento
              batch_size=64, #Tamano del lote 
              callbacks=[PlotLossesKeras(), parada_temprana], #Metemos la parada temprana y grafico a tiempo real del Learning curve por fold
              class_weight=diccionario_de_pesos, #Pesos para las clases
              verbose=1)
    
    histories.append(history.history) #Se agraga la historia de entrenamiento a las historias totales
    tf = time.time()
    duracion_xfold = tf- t0
    minutos_xfold = int(duracion_xfold // 60)
    segundos_xfold = int(duracion_xfold % 60)
    print(f"\nDuración entrenamiento en el Fold #{fold_idx+1}:\n {minutos_xfold} minutos y {segundos_xfold} segundos\n")
    
    ####################### Evaluacion con cv set#######################################
    
    #Se realiza la evaluación en validación (y luego se puede guardar métricas por fold)
    val_loss, val_acc = model.evaluate(X_val_fold, Y_val_fold, verbose=0)
    fold_accuracies.append(val_acc) #Se agrega la exactitud a la lista de exactitudes
    print(f"\nExactitud de la validación en el Fold #{fold_idx+1}: {val_acc:.4f}\n")
    
    y_pred_probs = model.predict(X_val_fold, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    #Se printea el reporte de classificacion de sklearn para ver las metricas y como funciona el modelo por fold
    print(f"\n Reporte de clasificación - Fold {fold_idx+1}\n")
    report = classification_report(Y_val_fold, y_pred, target_names=encoder.classes_, output_dict=True)
    fold_classification_reports.append(report) #Se agrega el reporte de clasificacion a la lista respectiva
    print(classification_report(Y_val_fold, y_pred, target_names=encoder.classes_))
    
    #Printeamos la matriz de confusion:
    umd.matriz_confusion(Y_val_fold, y_pred, encoder.classes_)

t_train_fin = time.time() #Tiempo final de entrenamiento
duracion = t_train_fin - t_train_inicio
minutos = int(duracion // 60)
segundos = int(duracion % 60)
print(f"\nDuración entrenamiento: {minutos} minutos y {segundos} segundos\n")



#################################### METRICAS DE EVALUACION GENERALES #############################################
#Printeamos las learning curves
umd.learning_curves(histories)

#Printeamos un reporte general de metricas:
umd.reporte_general_metricas(fold_accuracies, fold_classification_reports, encoder)

#Ploteo de analisis de varianza de los folds mediante los reportes de clasificacion:
umd.varianza_entre_folds(fold_classification_reports)

####################### Evaluacion con test set #######################################
#Se realiza la evaluación final en el conjunto de test (completamente independiente)
y_test_pred_probs = model.predict(X_test_final)
y_test_pred = np.argmax(y_test_pred_probs, axis=1)

# Matriz de confusión final
umd.matriz_confusion(Y_test, y_test_pred, labbels_clases=encoder.classes_)

# Reporte final
print("\n Reporte de Clasificación - Conjunto de Test")
print(classification_report(Y_test, y_test_pred, target_names=encoder.classes_))