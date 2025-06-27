"""
Este script implementa un pipeline de entrenamiento basado en Monte Carlo Cross-Validation
para una red neuronal multicapa (MLP), con el objetivo de encontrar el mejor modelo posible
a partir de múltiples inicializaciones aleatorias.

Resumen de flujo:
1. Carga y preprocesamiento del dataset (excluyendo 'OtroAGN')
2. Split estratificado en entrenamiento y prueba (80/20)
3. Aplicación de SMOTENC para balanceo de clases en el set de entrenamiento
4. Normalización de features continuas
5. Repetición de entrenamiento MLP por n_iteraciones (default: 100)
6. Filtro por métricas mínimas: F1 ponderado, F1 por clase, AUC total
7. Selección del mejor modelo entre los aceptados
8. Exportación del modelo en formatos .pkl y .h5, más un CSV con métricas
9. Visualización de matriz de confusión, curvas de pérdida, y clasificación final

Nota:
- Cada iteración es independiente y refleja la variabilidad inherente al entrenamiento de redes neuronales.
- Se recomienda aumentar `n_iteraciones` si se cuenta con suficiente tiempo de cómputo.
"""
# =================== CONFIGURACIONES ===================
import utils_model_dev as umd
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTENC
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle
import os
from datetime import datetime

# =================== CARGA Y PREPROCESADO ===================
#Se carga el archivo preprocesado con 3 clases predefinidas
X, Y, encoder = umd.cargar_dataset('df_final_clases_definidas_sin_OtroAGN.parquet', encoding='label', return_encoder=True)

#Se copian los datos y se evalua que esten en el formato indicado
X = X.copy()
Y = Y.astype(np.int32)
#Se designan las columnas para filtrar el la tecnica de SMOTENC de la forma correcta
cols_spectrum = [col for col in X.columns if col.startswith("spectrum_")]
cols_continuas = [col for col in X.columns if col not in cols_spectrum]
categorical_indices = [X.columns.get_loc(col) for col in cols_spectrum]

# =================== SPLIT FIJO ===================
#Split fijo de 80% training y 20% testing o validacion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# =================== SMOTENC ===================
#Se realiza la tecnica de data augmentation para las clases desbalanceadas (+300 en cada una) 
#para igualarlas en cantidad a la clase BLL
X_train_np = X_train.values
smote_nc = SMOTENC(categorical_features=categorical_indices,
                   sampling_strategy={1: 856, 2: 813},
                   random_state=42)
X_train_bal_np, Y_train_bal = smote_nc.fit_resample(X_train_np, Y_train)
X_train_bal = pd.DataFrame(X_train_bal_np, columns=X.columns)

# =================== ESCALADO ===================
#Se normalizan solo las columnas continuas. Las categóricas (spectrum) quedan tal cual.
scaler = StandardScaler()
X_train_bal[cols_continuas] = scaler.fit_transform(X_train_bal[cols_continuas])
X_test_scaled = X_test.copy()
X_test_scaled[cols_continuas] = scaler.transform(X_test[cols_continuas])

X_train_final = X_train_bal.values.astype(np.float32)
X_test_final = X_test_scaled.values.astype(np.float32)

# =================== ENTRENAMIENTO MONTE CARLO ===================
t_train_inicio = time.time()
#Se definen los parametros iniciales antes de entrenar
mejores_modelos = []
n_iteraciones = 100
umbral_f1_weighted = 0.85
umbral_f1_por_clase = {'BLL': 0.86, 'FSRQ': 0.81, 'NoAGN': 0.81}
umbral_auc_total = 0.94

for i in range(n_iteraciones):
    print(f"\n========== Entrenamiento Monte Carlo #{i+1} ==========")
    #Arquitectura fija del modelo
    model = umd.construir_modelo_dinamico(
        input_dim=X.shape[1],
        hidden_units=[121, 105, 137, 80],
        activaciones=['relu', 'selu', 'gelu', 'selu'],
        dropouts=[0.25, 0.15, 0.10, 0.10],
        output_units=len(encoder.classes_)
    )

    model.compile(optimizer=AdamW(learning_rate=0.000853497210084709),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #Tecnica de parada temprana
    parada_temprana = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    print("\nEntrenando modelo...")
    ti0 = time.time()
    
    #Se entrena el modelo
    history = model.fit(
        X_train_final, Y_train_bal,
        validation_split=0.2,
        epochs=1000,
        batch_size=55,
        callbacks=[parada_temprana],
        verbose=0
    )
    
    tif = time.time()
    print(f"\nDuración del entrenamiento: {int((tif - ti0) // 60)} min {int((tif - ti0) % 60)} s")
    
    #Se capturan los errores del entrenamiento:
    train_loss_final = history.history['loss'][-1]
    val_loss_final = history.history['val_loss'][-1]
    error_abs = abs(train_loss_final - val_loss_final)
    overfitting_ratio = val_loss_final / train_loss_final if train_loss_final != 0 else float('inf')
    print(f"Train loss final: {train_loss_final:.4f} | Val loss final: {val_loss_final:.4f} | Error absoluto: {error_abs:.6f} | Overfitting Ratio: {overfitting_ratio:.4f}")
    
    #Se realiza la prediccion con respecto al test set y se muestran las metricas
    y_pred = np.argmax(model.predict(X_test_final, verbose=0), axis=1)
    report = classification_report(Y_test, y_pred, target_names=encoder.classes_, output_dict=True)
    
    #Se guardan los f1 scores
    f1_weighted = report['weighted avg']['f1-score']
    f1_bll = report['BLL']['f1-score']
    f1_fsrq = report['FSRQ']['f1-score']
    f1_noagn = report['NoAGN']['f1-score']
    
    #Se calcula el auc score del modelo
    auc_total = roc_auc_score(tf.keras.utils.to_categorical(Y_test, num_classes=len(encoder.classes_)),
                              model.predict(X_test_final, verbose=0))
    
    #Se realiza un filtro de calidad para saber si el modelo que se entreno no cumplio las metricas
    if (
        f1_weighted >= umbral_f1_weighted and
        f1_bll >= umbral_f1_por_clase['BLL'] and
        f1_fsrq >= umbral_f1_por_clase['FSRQ'] and
        f1_noagn >= umbral_f1_por_clase['NoAGN'] and
        auc_total >= umbral_auc_total
    ):
        print(f"\u2714 Modelo aceptado con F1-weighted: {f1_weighted:.4f} | AUC: {auc_total:.4f}")
        mejores_modelos.append({
            'model': model,
            'scaler': scaler,
            'metrics': report,
            'f1_weighted': f1_weighted,
            'auc': auc_total,
            'history': history.history,
            'train_loss_final': train_loss_final,
            'val_loss_final': val_loss_final,
            'error_abs': error_abs,
            'overfitting_ratio': overfitting_ratio,
            'feature_names': X.columns.tolist()

        })
    else:
        print(f"\u274C Modelo descartado. No cumplió las métricas mínimas esperadas.")
        print(f"F1-weighted: {f1_weighted:.4f}, BLL: {f1_bll:.4f}, FSRQ: {f1_fsrq:.4f}, NoAGN: {f1_noagn:.4f}, AUC: {auc_total:.4f}")

fin = time.time()
print(f"\nTiempo total: {int((fin - t_train_inicio) // 60)} min {int((fin - t_train_inicio) % 60)} s")

# =================== SELECCION DEL MEJOR ===================
if mejores_modelos:
    mejor_modelo = max(mejores_modelos, key=lambda x: x['f1_weighted'])
    
    #Se exportan el mejor modelo ysus metricas para su estudio
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(directorio_base, '..', '..', 'data', 'monte carlo results')
    os.makedirs(folder, exist_ok=True)

    filename_pkl = os.path.join(folder, f"mejor_modelo_montecarlo_{timestamp}.pkl")
    with open(filename_pkl, 'wb') as f:
        pickle.dump(mejor_modelo, f)

    # Guardar el modelo en formato .h5
    filename_h5 = os.path.join(folder, f"mejor_modelo_montecarlo_{timestamp}.h5")
    mejor_modelo['model'].save(filename_h5)

    # Guardar reporte en .csv
    df_metrics = pd.DataFrame(mejor_modelo['metrics']).transpose()
    filename_csv = os.path.join(folder, f"reporte_metricas_montecarlo_{timestamp}.csv")
    df_metrics.to_csv(filename_csv)

    print(f"\n\u2728 Mejor modelo guardado en:")
    print(f"- PKL: {filename_pkl}")
    print(f"- H5: {filename_h5}")
    print(f"- Reporte CSV: {filename_csv}")
    print(f"F1-weighted: {mejor_modelo['f1_weighted']:.4f} | AUC: {mejor_modelo['auc']:.4f}")
    print(f"Train loss final: {mejor_modelo['train_loss_final']:.4f} | Val loss final: {mejor_modelo['val_loss_final']:.4f} | Error absoluto: {mejor_modelo['error_abs']:.6f} | Overfitting Ratio: {mejor_modelo['overfitting_ratio']:.4f}")

    # Graficar matriz de confusión final
    y_pred_final = np.argmax(mejor_modelo['model'].predict(X_test_final, verbose=0), axis=1)
    umd.matriz_confusion(Y_test, y_pred_final, encoder.classes_)

    # Imprimir reporte de métricas detallado
    print("\nReporte de clasificación (mejor modelo):")
    print(classification_report(Y_test, y_pred_final, target_names=encoder.classes_))

    # Graficar curvas de aprendizaje del mejor modelo
    umd.learning_curves([mejor_modelo['history']])
else:
    print("\nNo se encontró ningún modelo que superara los umbrales establecidos")
