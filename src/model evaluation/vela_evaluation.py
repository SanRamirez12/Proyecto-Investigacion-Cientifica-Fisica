# =================== Evaluación del modelo sobre fuentes de Vela ===================
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import utils_model_ev as ume
import pickle

# ======== CONFIGURACIÓN ========
directorio_actual = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(directorio_actual, '..', '..'))

path_modelo = os.path.join(base_dir, 'data', 'monte carlo results', 'mejor_modelo_montecarlo_2025-06-27_14-28-38.h5')
path_pkl = os.path.join(base_dir, 'data', 'monte carlo results', 'mejor_modelo_montecarlo_2025-06-27_14-28-38.pkl')
path_fuentes_vela = os.path.join(base_dir, 'data', 'post preliminary analysis', 'fuentes_vela.parquet')

# ======== CARGA DE MODELO Y DATOS ========
model = load_model(path_modelo)
with open(path_pkl, 'rb') as f:
    mejor_modelo = pickle.load(f)
scaler = mejor_modelo['scaler']

# Cargar dataset original para comparación visual
X_total, Y_total, encoder = ume.cargar_dataset('df_final_clases_definidas_sin_OtroAGN.parquet', encoding='label', return_encoder=True)
X_total['CLASS1'] = encoder.inverse_transform(Y_total)
feature_names = X_total.columns.tolist()
feature_names.remove('CLASS1')

# Cargar Vela y etiquetar
df_vela = pd.read_parquet(path_fuentes_vela)
df_vela['CLASS1'] = 'SrcVela'

# # Unir ambos datasets
# X_total_vis = X_total.copy()
# df_vela_vis = df_vela.copy()

# # Asegurar columnas consistentes para visualización
# for col in feature_names:
#     if col not in X_total_vis.columns:
#         X_total_vis[col] = 0
#     if col not in df_vela_vis.columns:
#         df_vela_vis[col] = 0

# # Agregar columna de clase
# X_total_vis['CLASE'] = X_total_vis['CLASS1']
# df_vela_vis['CLASE'] = df_vela_vis['CLASS1']

# # Concatenar y visualizar
# df_comparativo = pd.concat([X_total_vis[feature_names + ['CLASE']], df_vela_vis[feature_names + ['CLASE']]], ignore_index=True)

# sampled = df_comparativo.sample(frac=0.4, random_state=1)
# sns.pairplot(sampled, hue='CLASE', diag_kind='kde', plot_kws={'alpha': 0.5}, diag_kws={'fill': True})
# plt.suptitle('Distribución de clases conocidas vs Vela', y=1.02)
# plt.tight_layout()
# plt.show()

# ======== PREPROCESAMIENTO ========
X_vela = df_vela.drop(columns=['CLASS1'], errors='ignore').copy()

# Determinar columnas categóricas y continuas
cols_spectrum = [col for col in X_vela.columns if col.startswith("spectrum_")]
cols_continuas = [col for col in X_vela.columns if col not in cols_spectrum]

# Asegurar que todas las columnas esperadas están presentes
spectrum_cols = ['spectrum_plsuperexpcutoff', 'spectrum_powerlaw', 'spectrum_logparabola']
for col in spectrum_cols:
    if col not in X_vela.columns:
        X_vela[col] = 0

# Reordenar columnas según el entrenamiento
X_vela = X_vela[feature_names]

# Escalar columnas continuas
X_vela_scaled = X_vela.copy()
cols_continuas = [col for col in feature_names if not col.startswith("spectrum_")]
X_vela_scaled[cols_continuas] = scaler.transform(X_vela_scaled[cols_continuas])
X_vela_np = X_vela_scaled.values.astype(np.float32)

# ======== PREDICCIÓN ========
y_pred_probs = model.predict(X_vela_np, verbose=0)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# ======== DECODIFICACIÓN Y REPORTE ========
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

# ======== EXPORTACIÓN DETALLADA POR FUENTE ========
df_resultado = df_vela.copy()
df_resultado['CLASE_PREDICHA'] = clases  # etiquetas predichas
df_resultado['PROB_BLL'] = y_pred_probs[:, encoder.transform(['BLL'])[0]]
df_resultado['PROB_FSRQ'] = y_pred_probs[:, encoder.transform(['FSRQ'])[0]]
df_resultado['PROB_NoAGN'] = y_pred_probs[:, encoder.transform(['NoAGN'])[0]]

# Crear carpeta si no existe
output_dir = os.path.join(base_dir, 'data', 'model evaluation')
os.makedirs(output_dir, exist_ok=True)

# Guardar archivo
output_path = os.path.join(output_dir, 'predicciones_vela.csv')
df_resultado.to_csv(output_path, index=False)
print(f"\nArchivo 'predicciones_vela.csv' generado exitosamente en:\n{output_path}")


