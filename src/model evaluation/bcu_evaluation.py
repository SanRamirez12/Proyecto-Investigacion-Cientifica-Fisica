"""
Created on Mon Jun 23 18:56:05 2025

@author: Popi1
"""
# =================== Evaluación del modelo sobre fuentes BCU ===================
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import utils_model_dev as umd
import pickle

# ======== CONFIGURACIÓN ========
#Recordar cambiar estos archivos antes de empezar:
path_modelo = 'data/monte carlo results/mejor_modelo_montecarlo_YYYY-MM-DD_HH-MM-SS.h5'
path_pkl = 'data/monte carlo results/mejor_modelo_montecarlo_YYYY-MM-DD_HH-MM-SS.pkl'
path_fuentes_bcu = 'data/post preliminary analysis/fuentes_bcu.parquet'

# ======== CARGA DE MODELO Y DATOS ========
model = load_model(path_modelo)
with open(path_pkl, 'rb') as f:
    mejor_modelo = pickle.load(f)
scaler = mejor_modelo['scaler']
df_bcu = pd.read_parquet(path_fuentes_bcu)
X_bcu = df_bcu.copy()

# ======== PREPROCESAMIENTO ========
cols_spectrum = [col for col in X_bcu.columns if col.startswith("spectrum_")]
cols_continuas = [col for col in X_bcu.columns if col not in cols_spectrum]
X_bcu[cols_continuas] = scaler.transform(X_bcu[cols_continuas])
X_bcu_np = X_bcu.values.astype(np.float32)

# ======== PREDICCIÓN ========
y_pred_probs = model.predict(X_bcu_np, verbose=0)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# ======== DECODIFICACIÓN Y REPORTE ========
encoder = umd.cargar_dataset('df_final_clases_definidas_sin_OtroAGN.parquet', return_encoder=True)[2]
clases = encoder.inverse_transform(y_pred_labels)
conteo = pd.Series(clases).value_counts()

print("\nDistribución de clases predichas para fuentes BCU:")
print(conteo)

# ======== EVALUACIÓN FÍSICA (hipótesis: BCU debería ser BLL o FSRQ) ========
predicciones_noagn = np.sum(clases == 'NoAGN')
total_bcu = len(clases)
porcentaje_misclasificados = (predicciones_noagn / total_bcu) * 100

print(f"\nFuentes BCU clasificadas como NoAGN: {predicciones_noagn}/{total_bcu}")
print(f"Porcentaje de posible mala clasificación (NoAGN): {porcentaje_misclasificados:.2f}%")

# ======== EXPORTACIÓN OPCIONAL ========
df_resultado = df_bcu.copy()
df_resultado['CLASE_PREDICHA'] = clases
df_resultado.to_csv('data/model evaluation/predicciones_bcu.csv', index=False)
print("\nArchivo 'predicciones_bcu.csv' generado exitosamente.")

