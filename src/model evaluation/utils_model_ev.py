import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


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
