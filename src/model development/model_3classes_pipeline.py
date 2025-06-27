'''
Pipeline principal para la clasificacion de fuentes en 3 clases: 
    -BLL
    -FSRQ
    -NoAGN
Las mejores metricas alcanzadas en general fueron: 
Reporte general de métricas de entrenamiento: 10 min 21 s
Exactitud promedio entre folds: 85.43 ± 1.38%
F1-score promedio por clase entre folds:
 - BLL: 0.8802 ± 0.0122
 - FSRQ: 0.8260 ± 0.0200
 - NoAGN: 0.8315 ± 0.0205
F1-score (weighted) promedio entre folds: 0.8537 ± 0.0137
AUC promedio por clase en test:
BLL: 0.9428 ± 0.0080
FSRQ: 0.9445 ± 0.0092
NoAGN: 0.9563 ± 0.0056

'''

#Librerias de utils, y otras importantes:
import utils_model_dev as umd
import time
import numpy as np
import pandas as pd
from livelossplot import PlotLossesKeras

#Metodos de Skelearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
# from sklearn.utils.class_weight import compute_class_weight

#Metodos de Tensorflow
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping

#Metodos para la data augmentation
from imblearn.over_sampling import SMOTENC

#Empezamos leyendo los archivos. tomamos el datos sin fuentes no asociadas:
X, Y, encoder = umd.cargar_dataset('df_final_clases_definidas_sin_OtroAGN.parquet', encoding='label', return_encoder=True)
# print(f'Features: {X} y target Values: {Y}')

if 3 in Y:  # Suponiendo que OtroAGN era clase 3
    print("¡Advertencia! Se encontró la clase OtroAGN en los datos.")
    
#Se aseguran que son tipos correctos
X = X.copy()
Y = Y.astype(np.int32)
#print(f'Valores de X: \n{X}\n, Y: \n{Y}')

#Se identifican las columnas a no escalar (spectrum types) y las demas se escalan 
cols_spectrum = [col for col in X.columns if col.startswith("spectrum_")]
cols_continuas = [col for col in X.columns if col not in cols_spectrum]

# Indices de columnas categóricas (requerido por SMOTENC)
categorical_indices = [X.columns.get_loc(col) for col in cols_spectrum]

#Se define el K-Fold para el cross validation, en este caso de estratificado  y barajado para el imbalance de clases 
kfold = StratifiedShuffleSplit(
    n_splits=10, #Numero de folds
    test_size=0.15, #Divide el test set en 15% y es diferente para cada split
    random_state=69 #Semilla de generador de numero aleatorio
    )

#Inicializacion de estructuras para guardar resultados:
t_train_inicio = time.time() #Se define un tiempo inicial para entrenamiento  
fold_results = []

#################################### INICIO DE LOOP ##########################################
##############################################################################################
for fold_idx, (trainval_idx, test_idx) in enumerate(kfold.split(X, Y)):
    print(f"\n########## Fold externo #{fold_idx + 1} ##########")

    #Se define un test set a partir de los 15%  asignados en el SSS
    X_trainval, X_test = X.iloc[trainval_idx].copy(), X.iloc[test_idx].copy()
    Y_trainval, Y_test = Y[trainval_idx], Y[test_idx]

    #Se realiza una division de los sets train y val internos para cada fold
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=0.2, stratify=Y_trainval, random_state=fold_idx)
    
    # Antes de SMOTENC
    print("Distribución antes del balanceo:")
    print(pd.Series(Y_train).value_counts())
    
    #Usamos tecnica de data augmentation con SMOTENC
    X_train_np = X_train.values #convertimos el trainset en un np array
    smote_nc = SMOTENC(categorical_features=categorical_indices, #determinamos cuales son las columnas categoricas
                       sampling_strategy={1: 856, 2: 813}, #Estrategia de generacion de datros sinteticos
                       random_state=fold_idx)
    X_train_bal_np, Y_train_bal = smote_nc.fit_resample(X_train_np, Y_train)
    X_train_bal = pd.DataFrame(X_train_bal_np, columns=X.columns) # Volver a DataFrame para escalar
    
    # Después de SMOTENC
    print("Distribución después del balanceo:")
    print(pd.Series(Y_train_bal).value_counts())
    
    #Se define el scaler a usar:
    scaler = StandardScaler()
    #Se realizan copias de los sets para transformar y ajustar segun corresponda
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    #Se escalan los datos:
    X_train_bal[cols_continuas] = scaler.fit_transform(X_train_bal[cols_continuas])
    X_val_scaled[cols_continuas] = scaler.transform(X_val[cols_continuas])
    X_test_scaled[cols_continuas] = scaler.transform(X_test[cols_continuas])
    
    #Se verifica que sean del tipo deseado
    X_train_final = X_train_bal.values.astype(np.float32)
    X_val_final = X_val_scaled.values.astype(np.float32)
    X_test_final = X_test_scaled.values.astype(np.float32)

    #Se define el diccionario de Pesos de clase optimizados:
    
    # clases_unicas = np.unique(Y_train) #De acuerdo con labels de Y
    # pesos_balanceados = compute_class_weight(class_weight='balanced', classes=clases_unicas, y=Y_train)
    # diccionario_de_pesos = {clase: peso for clase, peso in zip(clases_unicas, pesos_balanceados)}
    
    # diccionario_de_pesos = {}
    # for i, clase in enumerate(encoder.classes_):
    #     if clase == 'BLL':
    #         diccionario_de_pesos[i] = 0.9267434224754194
    #     elif clase == 'FSRQ':
    #         diccionario_de_pesos[i] = 0.912883422297037
    #     elif clase == 'NoAGN':
    #         diccionario_de_pesos[i] = 1.8520622682254024

    #Se define la Arquitectura del Perceptron
    hidden_units = [121, 105, 137, 80] #Neuronas
    hidden_act_funct = ['relu', 'selu', 'gelu', 'selu'] #Activaciones
    tasa_abandono = [0.25, 0.15,  0.10, 0.10] #Dropout rates
    n_clases = len(encoder.classes_) #Cantidad de clases
    input_dim = X.shape[1]
    
    #Se construye el modelo
    model = umd.construir_modelo_dinamico(input_dim, hidden_units, hidden_act_funct, tasa_abandono, n_clases)
    
    #Se define el optimizador, funcion de costo, metrica y tasa de aprendizaje
    optimizador = AdamW(learning_rate= 0.000853497210084709) 
    model.compile(optimizer=optimizador, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    #Se hace un Callback de con parada anticipada en caso de que el error de validacion no mejore
    parada_temprana = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    ####################### Entrenamiento ####################################
    print("\nEntrenando modelo...")
    t0 = time.time()
    history = model.fit(
        X_train_final, Y_train_bal,
        validation_data=(X_val_final, Y_val),
        epochs=1000,
        batch_size=55,
        callbacks=[PlotLossesKeras(), parada_temprana],
        #class_weight=diccionario_de_pesos,
        verbose=1
    )
    tf = time.time()
    duracion = tf - t0
    print(f"\nDuración del entrenamiento: {int(duracion // 60)} min {int(duracion % 60)} s")
    
    ####################### Evaluaciones #######################################
    #Se evalua en validacion
    val_loss, val_acc = model.evaluate(X_val_final, Y_val, verbose=0)
    y_val_pred = np.argmax(model.predict(X_val_final, verbose=0), axis=1)
    val_report = classification_report(Y_val, y_val_pred, target_names=encoder.classes_, output_dict=True)

    #Se evalua en test
    y_test_probs = model.predict(X_test_final, verbose=0)
    y_test_pred = np.argmax(y_test_probs, axis=1)
    test_report = classification_report(Y_test, y_test_pred, target_names=encoder.classes_, output_dict=True)
    
    #Se printea el reporte de classificacion de sklearn para ver las metricas y como funciona el modelo por fold
    print(f"\nReporte en test - Fold {fold_idx + 1}\n")
    print(classification_report(Y_test, y_test_pred, target_names=encoder.classes_))
    umd.matriz_confusion(Y_test, y_test_pred, encoder.classes_)
    umd.graficar_roc_auc_multiclase(Y_test, y_test_probs, encoder, title=f"Curva ROC - Fold {fold_idx + 1}")
    
    #Se calcula el valor AUC por clase por fold
    y_true_bin = label_binarize(Y_test, classes=range(len(encoder.classes_)))
    auc_scores_fold = {}
    for i, clase in enumerate(encoder.classes_):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_test_probs[:, i])
        score = auc(fpr, tpr)
        auc_scores_fold[clase] = score
    
    #Se guardan los resultados del entrenaimiento de este fold
    fold_results.append({
        'model': model,
        'scaler': scaler,
        'history': history.history,
        'val_report': val_report,
        'test_report': test_report,
        'test_indices': test_idx,
        'auc_scores': auc_scores_fold
    })
#################################### FIN DE LOOP #############################################
##############################################################################################

#Se calcula el tiempo total de entrenamiento
fin = time.time()
print(f"\nTiempo total: {int((fin - t_train_inicio) // 60)} min {int((fin - t_train_inicio) % 60)} s")

#Se guardan los resultados del entrenamiento:
umd.guardar_resultados_entrenamiento(fold_results)

#Listas para las visualizaciones adicionales: 
test_reports = [r['test_report'] for r in fold_results]
test_accuracies = [r['test_report']['accuracy'] for r in fold_results]
histories = [r['history'] for r in fold_results]

#Visualizaciones adicionales:
umd.learning_curves(histories)
umd.reporte_general_metricas(test_accuracies, test_reports, encoder)
umd.varianza_entre_folds(test_reports)

#Comparacion de AUC por clase despues de todo el entrenamiento
print("\nAUC promedio por clase en test:")
all_auc_scores = {clase: [] for clase in encoder.classes_}
for r in fold_results:
    for clase in encoder.classes_:
        all_auc_scores[clase].append(r['auc_scores'][clase])
for clase in encoder.classes_:
    media = np.mean(all_auc_scores[clase])
    std = np.std(all_auc_scores[clase])
    print(f"{clase}: {media:.4f} ± {std:.4f}")






















