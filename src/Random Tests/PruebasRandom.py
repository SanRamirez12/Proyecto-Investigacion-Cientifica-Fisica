from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.impute import SimpleImputer

#Funcion que me lee el archivo actual 
def read_RawFile(nombre_archivo):
    #Se designa la ruta actual de este archivo:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta relativa al archivo .fits ubicado en ../data/raw/
    file_path = os.path.join(directorio_actual, '..', 'data', 'raw', nombre_archivo)
    return file_path

#definimos una funcion que plotea los histogramas dependiendo del parametro que le pasemos
def definirHistogramaClases(nombre_archivo, n_feature, nombre_asignado):
    #Creamos el path de la carpeta raw 
    path_archivo = read_RawFile(nombre_archivo)
    
    #Se asgina un feature del catalogo, solo para leer
    datosFits_4FGL = fits.open(name=path_archivo,mode="readonly")
    
    #Se accede al segundo elemento (índice 1) del archivo FITS
    catalogo_fuentes = datosFits_4FGL[1].data
    
    #Se extrae la columna correspondiente al parámetro n_feature de la tabla de datos
    columna = catalogo_fuentes[n_feature] #Preferiblemente clase
    
    #La columna extraída se convierte en un DataFrame de Pandas para facilitar su manipulación y análisis.
    columna_dataframe = pd.DataFrame(columna)
    
    #Se renombra la columna del DataFrame, cambiando el nombre predeterminado (0) por nombre_asignado.
    columna_dataframe.rename(columns={0: nombre_asignado}, inplace=True)
    
    #se realiza el ploteo del histograma de frecuencias porcentuales para el feature elegido:
    frecuencias = 100*columna_dataframe[nombre_asignado].value_counts()/len(columna)
    #Se utiliza el método plot de Pandas para crear un gráfico de barras que representa las frecuencias porcentuales de cada valor.
    frecuencias.plot(kind='bar')
    #Se plotean los datos
    plt.xlabel(nombre_asignado)
    plt.ylabel('Frecuencia relativa (%)')
    plt.title('Histograma de Frecuencias Relativas Porcentuales')
    plt.show()
    
    # booba1 = catalogo_fuentes['Flux_Band']
    # booba2 = catalogo_fuentes['nuFnu_Band']
    # booba3 = catalogo_fuentes['Flux_History']
    # for i in range(1):
    #     print(f'Fuente {i+1}: Flux_Band: {len(booba1[i])}, \n nuFnu_Band:  {len(booba2[i])},\n Flux_History: {len(booba3[i])}')
    # return 

def count_infinities(df): #
    # Verificar si hay valores infinitos (positivos o negativos)
    inf_counts = (df.isin([np.inf, -np.inf])).sum()
    
    # Devolver el conteo de infinitos por columna
    return inf_counts


#Imputar valores faltantes para cols con %NaNs < 1%:
def imputar_nans(df):
    #Generamos una copia del dataframe
    df_copy = df.copy()
    
    #Se excluyen las columnas categoricas para la imputacion
    cols_excluir = ['SpectrumType', 'CLASS1']
    df_excluir = df_copy[cols_excluir]
    df_numericas = df_copy.drop(columns=cols_excluir)

    #Se imputan los Nans faltantes con el SimpleImputer(strategy) de sklearn
    #strategy: 'constant' con valor 0 que indica que no aplica, no un numero normal
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    df_imputado = pd.DataFrame(
        #aplica el imputer a las columnas numericas
        imputer.fit_transform(df_numericas), 
        #se mantenga el mismo nombre de columnas
        columns=df_numericas.columns, 
        #conserva el indice de las columnas originales
        index=df_numericas.index )

    #Se reconstruye el DataFrame con todas las columnas
    df_final = pd.concat([df_imputado, df_excluir], axis=1)

    #Se realiza una verificacion para ver que no haya mas Nans
    assert df_final.isnull().sum().sum() == 0, "Todavía hay NaNs en el DataFrame."

    return df_final