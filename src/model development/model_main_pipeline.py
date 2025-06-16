#Librerias de utils, y otras importantes:
import utils_model_dev as umd
import time
import numpy as np
from livelossplot import PlotLossesKeras

#Metodos de Skelearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, label_binarize

#Metodos de Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping



#Empezamos leyendo los archivos. tomamos el datos sin fuentes no asociadas:
X, Y, encoder = umd.cargar_dataset('df_final_solo_clases_definidas.parquet', encoding='label', return_encoder=True)
# print(f'Features: {X} y target Values: {Y}')

#Se aseguran que son tipos correctos
X = X.copy()
Y = Y.astype(np.int32)
n_clases = len(encoder.classes_) #Cantidad de clases
#print(f'Valores de X: \n{X}\n, Y: \n{Y}')

#Se identifican las columnas a no escalar (spectrum types) y las demas se escalan 
cols_no_escaladas = [col for col in X.columns if col.startswith("spectrum_")]
cols_a_escalar = [col for col in X.columns if col not in cols_no_escaladas]

#Se define el K-Fold para el cross validation, en este caso de estratificado  y barajado para el imbalance de clases 
kfold = StratifiedShuffleSplit(
    n_splits=5, #Numero de folds
    test_size=0.15, #Divide el test set en 15% y es diferente para cada split
    random_state=47 #Semilla de generador de numero aleatorio
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

    #Se define el scaler a usar:
    scaler = StandardScaler()
    #Se realizan copias de los sets para transformar y ajustar segun corresponda
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    #Se escalan los datos:
    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_val_scaled[cols_a_escalar] = scaler.transform(X_val[cols_a_escalar])
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])
    
    #Se verifica que sean del tipo deseado
    X_train_final = X_train_scaled.values.astype(np.float32)
    X_val_final = X_val_scaled.values.astype(np.float32)
    X_test_final = X_test_scaled.values.astype(np.float32)

    #Se define el diccionario de Pesos de clase balanceados
    clases_unicas = np.unique(Y_train) #De acuerdo con labels de Y
    pesos_balanceados = compute_class_weight(class_weight='balanced', classes=clases_unicas, y=Y_train)
    diccionario_de_pesos = {clase: peso for clase, peso in zip(clases_unicas, pesos_balanceados)}

    #Se define la Arquitectura del Perceptron
    hidden_units = [93, 97, 90]
    hidden_act_funct = ['tanh', 'relu', 'elu']
    tasa_abandono = [0.0512, 0.3426, 0.2291] #Redondeados a 4 decimales

    model = Sequential([
        Dense(hidden_units[0], input_shape=(X.shape[1],), activation=hidden_act_funct[0], name='hidden_layer1'),
        Dropout(tasa_abandono[0]),
        Dense(hidden_units[1], activation=hidden_act_funct[1], name='hidden_layer2'),
        Dropout(tasa_abandono[1]),
        Dense(hidden_units[2], activation=hidden_act_funct[2], name='hidden_layer3'),
        Dropout(tasa_abandono[2]),
        Dense(n_clases, activation='softmax', name='output_layer')
    ])
    
    #Se define el optimizador, funcion de costo, metrica y tasa de aprendizaje
    optimizador = RMSprop(learning_rate=0.0005) #Redondeado a 4 decimales
    model.compile(optimizer=optimizador, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    #Se hace un Callback de con parada anticipada en caso de que el error de validacion no mejore
    parada_temprana = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    ####################### Entrenamiento ####################################
    print("\nEntrenando modelo...")
    t0 = time.time()
    history = model.fit(
        X_train_final, Y_train,
        validation_data=(X_val_final, Y_val),
        epochs=1000,
        batch_size=49,
        callbacks=[PlotLossesKeras(), parada_temprana],
        class_weight=diccionario_de_pesos,
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






















