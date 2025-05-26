#Librerias de utils, y otras importantes:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#Metodos de Skelearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

#Ocupamos una funcion que nos lea archivos de datos .parquet o .csv y nos devuelva 
# separademente X e Y, y codifica la columna CLASS1.
def cargar_dataset(nombre_archivo, encoding='label', return_encoder=False):
    
    #Se designa la ruta actual de este archivo:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta relativa al archivo ubicado en ../data/post preliminary analysis/
    filepath = os.path.join(
        directorio_actual, '..', '..', 'data', 'post preliminary analysis', nombre_archivo)
    
    #Se carga el achivo con pandas dependiendo de su tipo
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("El archivo debe ser .parquet o .csv")

    #Se verifica que CLASS1 está en las columnas, para separar features de target
    if 'CLASS1' not in df.columns:
        raise ValueError("No se encontró la columna 'CLASS1' en el dataset")

    #Se separan X (features)  e Y (target values)
    X = df.drop(columns=['CLASS1'])
    y_raw = df['CLASS1']

    #Se verifica si el encoding es de label o onehot y se printean para ver sus nuevas demarcaciones:
    #Primero label enconding
    if encoding == 'label':
        #Se usa el encoder de sklearn:
        encoder = LabelEncoder()
        #Se transforman/codifican los target values
        Y = encoder.fit_transform(y_raw)
        print("LabelEncoder mapeo de clases:")
        for idx, label in enumerate(encoder.classes_):
            print(f"{idx}: {label}")

    else:
        #atrapa cualquier error en caso de cambiar el encoding
        raise ValueError("encoding debe ser 'label'")
    
    #Si funciono el encoding, devueleve los dataframes
    if return_encoder:
        return X, Y, encoder
    return X, Y

#Metodo que me plotea la matriz de confusion para ver los valores positivos y negativos/ falsos y verdaderos
def matriz_confusion(y_true, y_pred, labbels_clases):
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    #Realiza un heatmap de seaborn
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labbels_clases,
                yticklabels=labbels_clases)
    plt.xlabel('Clases Predichas')
    plt.ylabel('Clases Verdaderas')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()
    
#Metodo para graficar las curvas de aprendizaje.
def learning_curves(histories):
    plt.figure(figsize=(10, 4))
    
    for i, hist in enumerate(histories):
        plt.plot(hist['loss'], alpha=0.5, label=f'Fold {i+1} - Entrenamiento')
        plt.plot(hist['val_loss'], alpha=0.5, linestyle='--', label=f'Fold {i+1} - Validacion')
    
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Curvas de aprendizaje por fold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Escalamos datasets para mejor tratamiento en red neuronal
def escalar_datasets(X_temp, X_test, cols_a_escalar):
    #Se convierte el df a array NumPy (manteniendo el orden de columnas)
    X_temp_escalado = X_temp.copy()
    X_test_escalado = X_test.copy()
    scaler = StandardScaler() #Difinimos el escalador como un tipo StandarScaler
    X_temp_escalado[cols_a_escalar] = scaler.fit_transform(X_temp_escalado[cols_a_escalar]) #Se ajusta y escala X del training/vsalidation
    X_test_escalado[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar]) #Se escala el test set
     #.values cambia df a narray, y tipo float32 tiene mejor rendimiento que el float64
    X_temp_final = X_temp_escalado.values.astype(np.float32) 
    X_test_final = X_test_escalado.values.astype(np.float32) 
    return X_temp_final, X_test_final

#Metodo de reporte general de metricas: 
def reporte_general_metricas(fold_accuracies, fold_classification_reports, encoder):
    print('\nReporte general de métricas de entrenamiento: \n')
    #Exactitud promedio promedio y desviación estándar
    acc_mean = np.mean(fold_accuracies)
    acc_std = np.std(fold_accuracies)
    print(f"\n Exactitud promedio entre folds: {acc_mean:.4f} ± {acc_std:.4f}\n")

    #F1-score promedio por clase
    f1_scores = {class_name: [] for class_name in encoder.classes_}

    for report in fold_classification_reports:
        for class_name in encoder.classes_:
            f1_scores[class_name].append(report[class_name]["f1-score"])

    print("\n F1-score promedio por clase entre folds:\n")
    for class_name in encoder.classes_:
        f1_mean = np.mean(f1_scores[class_name])
        f1_std = np.std(f1_scores[class_name])
        print(f" - {class_name}: {f1_mean:.4f} ± {f1_std:.4f}")

#Metodo que genera un boxplot de la distribución del F1-score por clase entre folds.
def varianza_entre_folds(fold_classification_reports):
    if not fold_classification_reports:
        print("No se proporcionaron reportes de clasificación.")
        return

    #Se extraen los nombres de clases (descarta métricas agregadas como 'accuracy', 'macro avg', etc.)
    clases = [k for k in fold_classification_reports[0].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

    #Se inicializa listas de F1 por clase
    f1_scores = {clase: [] for clase in clases}

    for report in fold_classification_reports:
        for clase in clases:
            f1_scores[clase].append(report[clase]['f1-score'])

    #Se plotea el boxplot: 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[f1_scores[clase] for clase in clases])
    plt.xticks(ticks=range(len(clases)), labels=clases)
    plt.ylabel('F1-score')
    plt.title('Distribución de F1-score por clase entre folds')
    plt.grid(True)
    plt.tight_layout()
    plt.show()







