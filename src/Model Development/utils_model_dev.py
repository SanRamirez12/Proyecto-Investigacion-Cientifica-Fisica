import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    plt.ylabel('Clases Verdaders')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()
    
#Metodo para graficar las curvas de aprendizaje.
def learning_curves(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Pérdida del training set')
    plt.plot(history.history['val_loss'], label='Pérdida del validation set')
    plt.xlabel('Epochs(Épocas)')
    plt.ylabel('Loss (Pérdida)')
    plt.title('Curvas de aprendizaje')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()