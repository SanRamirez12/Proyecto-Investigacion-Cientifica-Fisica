"""
Created on Mon Jun 23 18:56:19 2025

@author: Popi1
"""
# =================== Evaluación del modelo sobre fuentes de Vela ===================
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import utils_model_dev as umd
import pickle

# ======== CONFIGURACIÓN ========
path_modelo = 'data/monte carlo results/mejor_modelo_montecarlo_YYYY-MM-DD_HH-MM-SS.h5'
path_pkl = 'data/monte carlo results/mejor_modelo_montecarlo_YYYY-MM-DD_HH-MM-SS.pkl'
path_fuentes_vela = 'data/post preliminary analysis/fuentes_vela.parquet'

# ======== CARGA DE MODELO Y DATOS ========
model = load_model(path_modelo)
with open(path_pkl, 'rb') as f:
    mejor_modelo = pickle.load(f)
scaler = mejor_modelo['scaler']
df_vela = pd.read_parquet(path_fuentes_vela)
X_vela = df_vela.copy()

# ======== PREPROCESAMIENTO ========
cols_spectrum = [col for col in X_vela.columns if col.startswith("spectrum_")]
cols_continuas = [col for col in X_vela.columns if col not in cols_spectrum]
X_vela[cols_continuas] = scaler.transform(X_vela[cols_continuas])
X_vela_np = X_vela.values.astype(np.float32)

# ======== PREDICCIÓN ========
y_pred_probs = model.predict(X_vela_np, verbose=0)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# ======== DECODIFICACIÓN Y REPORTE ========
encoder = umd.cargar_dataset('df_final_clases_definidas_sin_OtroAGN.parquet', return_encoder=True)[2]
clases = encoder.inverse_transform(y_pred_labels)
conteo = pd.Series(clases).value_counts()

print("\nDistribución de clases predichas para fuentes de la región de Vela:")
for clase, cantidad in conteo.items():
    print(f"{clase}: {cantidad} fuentes")

total_vela = len(clases)
predicciones_noagn = np.sum(clases == 'NoAGN')
porcentaje_noagn = (predicciones_noagn / total_vela) * 100

print(f"\nTotal de fuentes: {total_vela}")
print(f"Fuentes Vela clasificadas como NoAGN: {predicciones_noagn}/{total_vela}")
print(f"Porcentaje clasificado como NoAGN (esperado alto): {porcentaje_noagn:.2f}%")

# ======== EXPORTACIÓN OPCIONAL ========
df_resultado = df_vela.copy()
df_resultado['CLASE_PREDICHA'] = clases
df_resultado.to_csv('data/model evaluation/predicciones_vela.csv', index=False)
print("\nArchivo 'predicciones_vela.csv' generado exitosamente.")
