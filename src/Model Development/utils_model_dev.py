#Librerias de utils, y otras importantes:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
from datetime import datetime

#Metodos de Skelearn
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc

#Metodos de TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

#Metodo que construye un modelo MLP dinámicamente según las listas dadas.
def construir_modelo_dinamico(input_dim, hidden_units, activaciones, dropouts, output_units):
    #Se verifica que las listas esten en la misma dimension
    assert len(hidden_units) == len(activaciones) == len(dropouts), "Longitudes de configuración inconsistente"
    #Se establece el modelo de red
    model = Sequential()
    #Se crea el modelo de acuerdo al numero de capas
    for i in range(len(hidden_units)):
        if i == 0: #Para el hidden layer 1 se establece el inputdim
            model.add(Dense(hidden_units[i], input_dim=input_dim, activation=activaciones[i], name=f"hidden_layer{i+1}"))
        else: #Para las capas ocultas 
            model.add(Dense(hidden_units[i], activation=activaciones[i], name=f"hidden_layer{i+1}"))
        
        model.add(Dropout(dropouts[i])) #Se establecen los dropout layer despues de cada hidden layer
    #Se establece el output layer para el numero de clases y con activacion softmax.
    model.add(Dense(output_units, activation='softmax', name='output_layer'))
    #Se retorna elmodelo
    return model

######################################## METODOS DE VISUALIZACIONES Y METRICAS #####################################################
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
   fig, ax = plt.subplots(figsize=(10, 5))

   lines = []  # para recolectar handles
   labels = []  # para recolectar etiquetas

   for i, hist in enumerate(histories):
       l1, = ax.plot(hist['loss'], alpha=0.5, label=f'Fold {i+1} - Entrenamiento')
       l2, = ax.plot(hist['val_loss'], alpha=0.5, linestyle='--', label=f'Fold {i+1} - Validación')
       lines.extend([l1, l2])
       labels.extend([f'Fold {i+1} - Entrenamiento', f'Fold {i+1} - Validación'])

   ax.set_xlabel('Épocas')
   ax.set_ylabel('Pérdida')
   ax.set_title('Curvas de aprendizaje por fold')
   ax.grid(True)

   # Crear figura adicional para leyenda
   fig.subplots_adjust(bottom=0.35)  # Deja espacio suficiente
   fig.legend(lines, labels,
                       loc='upper center', bbox_to_anchor=(0.5, 0),
                       ncol=4, fancybox=True, shadow=True)

   plt.tight_layout()
   plt.show()

#Metodo de reporte general de metricas: 
def reporte_general_metricas(fold_accuracies, fold_classification_reports, encoder):
    print('\nReporte general de métricas de entrenamiento: \n')
    #Exactitud promedio promedio y desviación estándar
    acc_mean = np.mean(fold_accuracies)
    acc_std = np.std(fold_accuracies)
    print(f"\n Exactitud promedio entre folds: {acc_mean:.4f} ± {acc_std:.4f}\n")

    #F1-score promedio por clase
    f1_scores = {class_name: [] for class_name in encoder.classes_}
    f1_weighted = []
    
    for report in fold_classification_reports:
        for class_name in encoder.classes_:
            f1_scores[class_name].append(report[class_name]["f1-score"])
        # Guardamos también el f1_score promedio ponderado (weighted)
        f1_weighted.append(report["weighted avg"]["f1-score"])

    print("\n F1-score promedio por clase entre folds:\n")
    for class_name in encoder.classes_:
        f1_mean = np.mean(f1_scores[class_name])
        f1_std = np.std(f1_scores[class_name])
        print(f" - {class_name}: {f1_mean:.4f} ± {f1_std:.4f}")
        
    # F1-score weighted global entre folds
    f1_w_mean = np.mean(f1_weighted)
    f1_w_std = np.std(f1_weighted)
    print(f"\n F1-score (weighted) promedio entre folds: {f1_w_mean:.4f} ± {f1_w_std:.4f}")

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

#Metodo para mostrar los pesos por fold:
def mostrar_pesos_fold(pesos_por_fold, encoder):
    print("\nResumen de pesos de clase por fold:")
    
    #Se inicilizala estructura para acumular pesos por clase
    acumulador = {clase: [] for clase in encoder.transform(encoder.classes_)}
    
    for i, pesos in enumerate(pesos_por_fold, 1):
        print(f"\nFold #{i}:")
        for clase_idx, peso in pesos.items():
            clase_nombre = encoder.inverse_transform([clase_idx])[0]
            print(f"  Clase {clase_idx} ({clase_nombre}): {peso:.4f}")
            acumulador[clase_idx].append(peso)
    
    print("\nPromedio de pesos por clase a lo largo de los folds:")
    for clase_idx in sorted(acumulador.keys()):
        clase_nombre = encoder.inverse_transform([clase_idx])[0]
        promedio = np.mean(acumulador[clase_idx])
        print(f"  Clase {clase_idx} ({clase_nombre}): {promedio:.4f}")

#Metodo para graficar pesos por clase
def graficar_pesos_por_fold(pesos_por_fold, encoder):

    clases_idx = sorted(encoder.transform(encoder.classes_))

    #Se crea una estructura por clase
    pesos_por_clase = {clase_idx: [] for clase_idx in clases_idx}
    
    for fold_pesos in pesos_por_fold:
        for clase_idx in clases_idx:
            peso = fold_pesos.get(clase_idx, np.nan)  # Evita errores si falta alguna clase
            pesos_por_clase[clase_idx].append(peso)

    #Se grafica
    plt.figure(figsize=(10, 6))
    for clase_idx in clases_idx:
        pesos = pesos_por_clase[clase_idx]
        nombre = encoder.inverse_transform([clase_idx])[0]
        promedio = np.mean(pesos)
        plt.plot(range(1, len(pesos) + 1), pesos, marker='o', label=f'{nombre}')
        plt.hlines(promedio, 1, len(pesos), linestyles='dashed', colors='gray', alpha=0.5)

    plt.xlabel('Fold')
    plt.ylabel('Peso de clase')
    plt.title('Pesos de clase por fold (con promedio)')
    plt.xticks(range(1, len(pesos_por_fold) + 1))
    plt.legend(title='Clases')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

#Metodo para graficar las curvas Roc y Auc de las clases
def graficar_roc_auc_multiclase(y_true, y_pred_probs, encoder, title='Curvas ROC por clase'):
    clases = encoder.classes_
    n_classes = len(clases)

    #Convierte las clases verdaderas en formato binario (uno contra el resto)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    #Se definen Curvas ROC y AUC por clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    #Calcula la Tasa de Verdaderos Positivos (TPR) y Falsos Positivos (FPR)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #Plotea una línea ROC por clase con su respectivo AUC: 
    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{clases[i]} (AUC = {roc_auc[i]:.2f})')
    
    #Se traza la diagonal de referencia (modelo aleatorio):
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Guardar fold results en un .pkl
def guardar_resultados_entrenamiento(resultados):
    if not isinstance(resultados, (dict, list)):
        raise ValueError("El objeto a guardar debe ser una lista o diccionario (ej. fold_results).")

    #Se construye el nombre del archivo con timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') #Year/month/day/hour/minute/second
    nombre_archivo = f"fold_results_{timestamp}.pkl"

    #Se define la ruta base del script
    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    #Se define la carpeta de salida
    carpeta_salida = os.path.join(directorio_actual, '..', '..', 'data', 'fold results training')
    os.makedirs(carpeta_salida, exist_ok=True)

    #Se define la ruta final
    ruta_archivo = os.path.join(carpeta_salida, nombre_archivo)

    # Guardar archivo
    joblib.dump(resultados, ruta_archivo)
    print(f"Resultados de entrenamiento guardados en: {ruta_archivo}")