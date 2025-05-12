import utils_model_dev as umd
from sklearn.model_selection import train_test_split

#Empezamos leyendo los archivos. tomamos el datos sin fuentes no asociadas:
X, Y, encoder = umd.cargar_dataset('df_final_sin_UncAss.parquet', encoding='label', return_encoder=True)
# print(f'Features: {X} y target Values: {Y}')

#Dividir el dataset, se separa en train, val y test (60/20/20) con stratificación
X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)