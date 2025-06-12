'''
El pipeline es exactamente igual al anterior solamente que  en esta le aplicamos smote a la clase minoritaria, 
para ver si mejora la clasificacion de dicha clase. 
'''
# Librerías de utils, y otras importantes:
import utils_model_dev as umd
import time
import numpy as np
from livelossplot import PlotLossesKeras

# Métodos de Sklearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, label_binarize

# Métodos de Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# SMOTE para oversampling de clase minoritaria
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Empezamos leyendo los archivos:
X, Y, encoder = umd.cargar_dataset('df_final_sin_UncAss.parquet', encoding='label', return_encoder=True)
X = X.copy()
Y = Y.astype(np.int32)

cols_no_escaladas = [col for col in X.columns if col.startswith("spectrum_")]
cols_a_escalar = [col for col in X.columns if col not in cols_no_escaladas]

kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=47)

t_train_inicio = time.time()
fold_results = []

for fold_idx, (trainval_idx, test_idx) in enumerate(kfold.split(X, Y)):
    print(f"\n########## Fold externo #{fold_idx + 1} ##########")

    X_trainval, X_test = X.iloc[trainval_idx].copy(), X.iloc[test_idx].copy()
    Y_trainval, Y_test = Y[trainval_idx], Y[test_idx]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=0.2, stratify=Y_trainval, random_state=fold_idx)

    # Escalar antes de aplicar SMOTE (sólo a columnas numéricas)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
    X_val_scaled[cols_a_escalar] = scaler.transform(X_val[cols_a_escalar])
    X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])

    ############################   SMOTE    ################################
    #Aplicar SMOTE solo a la clase minoritaria (OtroAGN = clase 4)
    smote = SMOTE(sampling_strategy={4: 250}, random_state=42, k_neighbors=3)
    X_train_sm, Y_train_sm = smote.fit_resample(X_train_scaled, Y_train)
    ##########################################################################
    
    X_train_final = X_train_sm.values.astype(np.float32)
    X_val_final = X_val_scaled.values.astype(np.float32)
    X_test_final = X_test_scaled.values.astype(np.float32)

    clases_unicas = np.unique(Y_train_sm)
    pesos_balanceados = compute_class_weight(class_weight='balanced', classes=clases_unicas, y=Y_train_sm)
    diccionario_de_pesos = {clase: peso for clase, peso in zip(clases_unicas, pesos_balanceados)}

    hidden_units = [93, 97, 90]
    hidden_act_funct = ['tanh', 'relu', 'elu']
    tasa_abandono = [0.0512, 0.3426, 0.2291]

    model = Sequential([
        Dense(hidden_units[0], input_shape=(X.shape[1],), activation=hidden_act_funct[0]),
        Dropout(tasa_abandono[0]),
        Dense(hidden_units[1], activation=hidden_act_funct[1]),
        Dropout(tasa_abandono[1]),
        Dense(hidden_units[2], activation=hidden_act_funct[2]),
        Dropout(tasa_abandono[2]),
        Dense(5, activation='softmax')
    ])

    optimizador = RMSprop(learning_rate=0.0011)
    model.compile(optimizer=optimizador, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    parada_temprana = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    print("\nEntrenando modelo...")
    t0 = time.time()
    history = model.fit(
        X_train_final, Y_train_sm,
        validation_data=(X_val_final, Y_val),
        epochs=1000,
        batch_size=49,
        callbacks=[PlotLossesKeras(), parada_temprana],
        class_weight=diccionario_de_pesos,
        verbose=1
    )
    tf = time.time()
    print(f"\nDuración del entrenamiento: {int((tf - t0) // 60)} min {int((tf - t0) % 60)} s")

    val_loss, val_acc = model.evaluate(X_val_final, Y_val, verbose=0)
    y_val_pred = np.argmax(model.predict(X_val_final, verbose=0), axis=1)
    val_report = classification_report(Y_val, y_val_pred, target_names=encoder.classes_, output_dict=True)

    y_test_probs = model.predict(X_test_final, verbose=0)
    y_test_pred = np.argmax(y_test_probs, axis=1)
    test_report = classification_report(Y_test, y_test_pred, target_names=encoder.classes_, output_dict=True)

    print(f"\nReporte en test - Fold {fold_idx + 1}\n")
    print(classification_report(Y_test, y_test_pred, target_names=encoder.classes_))
    umd.matriz_confusion(Y_test, y_test_pred, encoder.classes_)
    umd.graficar_roc_auc_multiclase(Y_test, y_test_probs, encoder, title=f"Curva ROC - Fold {fold_idx + 1}")

    y_true_bin = label_binarize(Y_test, classes=range(len(encoder.classes_)))
    auc_scores_fold = {}
    for i, clase in enumerate(encoder.classes_):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_test_probs[:, i])
        score = auc(fpr, tpr)
        auc_scores_fold[clase] = score

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

fin = time.time()
print(f"\nTiempo total: {int((fin - t_train_inicio) // 60)} min {int((fin - t_train_inicio) % 60)} s")

test_reports = [r['test_report'] for r in fold_results]
test_accuracies = [r['test_report']['accuracy'] for r in fold_results]
histories = [r['history'] for r in fold_results]

umd.learning_curves(histories)
umd.reporte_general_metricas(test_accuracies, test_reports, encoder)
umd.varianza_entre_folds(test_reports)

print("\nAUC promedio por clase en test:")
all_auc_scores = {clase: [] for clase in encoder.classes_}
for r in fold_results:
    for clase in encoder.classes_:
        all_auc_scores[clase].append(r['auc_scores'][clase])
for clase in encoder.classes_:
    media = np.mean(all_auc_scores[clase])
    std = np.std(all_auc_scores[clase])
    print(f"{clase}: {media:.4f} ± {std:.4f}")

