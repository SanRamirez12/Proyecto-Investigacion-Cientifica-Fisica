# =================== Evaluación del modelo sobre fuentes BCU ===================
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
path_fuentes_bcu = os.path.join(base_dir, 'data', 'post preliminary analysis', 'df_final_solo_BCU.parquet')

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

# Cargar BCU y etiquetar
df_bcu = pd.read_parquet(path_fuentes_bcu)
df_bcu['CLASS1'] = 'BCU'

# Unir ambos datasets
X_total_vis = X_total.copy()
df_bcu_vis = df_bcu.copy()

# Asegurar columnas consistentes para visualización
for col in feature_names:
    if col not in X_total_vis.columns:
        X_total_vis[col] = 0
    if col not in df_bcu_vis.columns:
        df_bcu_vis[col] = 0

# Agregar columna de clase
X_total_vis['CLASE'] = X_total_vis['CLASS1']
df_bcu_vis['CLASE'] = df_bcu_vis['CLASS1']

# Concatenar y visualizar
df_comparativo = pd.concat([X_total_vis[feature_names + ['CLASE']], df_bcu_vis[feature_names + ['CLASE']]], ignore_index=True)

sampled = df_comparativo.sample(frac=0.4, random_state=1)
sns.pairplot(sampled, hue='CLASE', diag_kind='kde', plot_kws={'alpha': 0.5}, diag_kws={'fill': True})
plt.suptitle('Distribución de clases conocidas vs BCU', y=1.02)
plt.tight_layout()
plt.show()

# ======== PREPROCESAMIENTO ========
X_bcu = df_bcu.drop(columns=['CLASS1'], errors='ignore').copy()

# Determinar columnas categóricas y continuas
cols_spectrum = [col for col in X_bcu.columns if col.startswith("spectrum_")]
cols_continuas = [col for col in X_bcu.columns if col not in cols_spectrum]

# Asegurar que todas las columnas esperadas están presentes
spectrum_cols = ['spectrum_plsuperexpcutoff', 'spectrum_powerlaw', 'spectrum_logparabola']
for col in spectrum_cols:
    if col not in X_bcu.columns:
        X_bcu[col] = 0

# Reordenar columnas según el entrenamiento
X_bcu = X_bcu[feature_names]

# Escalar columnas continuas
X_bcu_scaled = X_bcu.copy()
cols_continuas = [col for col in feature_names if not col.startswith("spectrum_")]
X_bcu_scaled[cols_continuas] = scaler.transform(X_bcu_scaled[cols_continuas])
X_bcu_np = X_bcu_scaled.values.astype(np.float32)

# ======== PREDICCIÓN ========
y_pred_probs = model.predict(X_bcu_np, verbose=0)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# ======== DECODIFICACIÓN Y REPORTE ========
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



