#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from astropy.io import fits
from sklearn.preprocessing import StandardScaler

#Preseleccion de columnas (No incluye flujos por ahora.)
def select_columnas(headers):
    #Los nombres de columnas estan guardados en Ttypes del 1 al 79 dentro de headers, por tanto:
    ttypes = []
    for i in range(1, 80):  # Desde TTYPE1 hasta TTYPE79
        nombre_columna = f'TTYPE{i}'
        if nombre_columna in headers:
            ttypes.append(headers[nombre_columna])
    #print(ttypes) #Tipos de columnas
    #A partir de los nombres de columnas, los vemos y elegimos de los 79 los que nos parecen:
    # Los uncertanties de las energias no los tomamos.
    # Las columnas 'Flux_Band', 'nuFnu_Band', 'Flux_History' son arrays de 8, 8 y 14 valores respectivamentes:
    columnas_seleccionadas = [
        'Flux1000', 'Energy_Flux100', 'SpectrumType', 'PL_Flux_Density',
        'PL_Index', 'LP_Flux_Density', 'LP_Index', 'LP_beta', 'LP_SigCurv',
        'LP_EPeak', 'PLEC_Flux_Density', 'PLEC_IndexS', 'PLEC_ExpfactorS',
        'PLEC_Exp_Index', 'PLEC_SigCurv', 'PLEC_EPeak', 'Npred',
        'Variability_Index', 'Frac_Variability', 'Flux_Peak', 'CLASS1'
        ]
    return columnas_seleccionadas

#Leer el archivo y transformarlo como un data frame
def leer_fits(nombre_archivo):
    #Se designa la ruta actual de este archivo:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta relativa al archivo .fits ubicado en ../data/raw/
    file_path = os.path.join(directorio_actual, '..', 'data', 'raw', nombre_archivo)

    #Se asgina un feature del catalogo, solo para leer
    datos_fits_4fgl = fits.open(name=file_path, mode="readonly")
    #print(datosFits_4FGL.info()) #Tipos de datos
    
    #Se accede al segundo elemento (índice 1) del archivo FITS: LAT_Point_Source_Catalog
    catalogo_fuentes = datos_fits_4fgl[1].data
    
    #Visualizar info de headers:
    headers = datos_fits_4fgl[1].header
    
    #Se seleccionan las colmunas respectivas
    columnas_seleccionadas = select_columnas(headers)
    
    #Se crea un diccionario donde se filtran los datos por las columnas seleccionadas
    datos = {col: catalogo_fuentes[col] for col in columnas_seleccionadas}
    #Se crea el dataframe usando pandas.
    df = pd.DataFrame(datos)
    #print(df)
    return df

#Arreglo de labels en CLASS1 para trabajar con mas orden:
def limpiar_labels_clases(df):
   #Ciclo que se aplicará a cada valor de la columna CLASS1
    for index, row in df.iterrows():
        #Se normaliza el valor de 'CLASS1' a minúsculasy eliminaespcios para evitar problemas con mayúsculas
        class_value = str(row['CLASS1']).lower().strip()  
        
        if class_value == 'fsrq':
            df.at[index, 'CLASS1'] = 'FSRQ'
            
        elif class_value == 'bll':
            df.at[index, 'CLASS1'] = 'BLL'
            
        elif class_value == 'bcu':
            df.at[index, 'CLASS1'] = 'BCU'
            
        elif class_value in ['rdg', 'nlsy1', 'sey', 'agn', 'css', 'ssrq']:
            df.at[index, 'CLASS1'] = 'OtroAGN'
            
        elif class_value == '':
            df.at[index, 'CLASS1'] = 'UncAss'
            
        else:
            df.at[index, 'CLASS1'] = 'NoAGN'
    
    return df

#Verificacion de porcentajes de NaNs por columna.
def estudio_nans(df):
    #Cuenta la cantidad de NaNs del dataframe usando .isna(bool) y luego los suma
    total_nans = df.isna().sum()      
    #Promedia los True o False de NaNs y saca un porcentaje con respecto a los 7195 datos      
    porcentaje = df.isna().mean() * 100 
    
    #Se crea un dataframe con los valores que queremos de NaNs
    df_resumen_nans = pd.DataFrame({
        'Cantidad de NaNs:': total_nans,
        'Porcentaje de NaNs:': porcentaje
    })
    
    #Muestra solo las columnas con NaNs:
    df_resumen_nans = df_resumen_nans[df_resumen_nans['Cantidad de NaNs:'] > 0]  
    #Ordena las columnas de mayor a menor:
    df_resumen_nans = df_resumen_nans.sort_values('Porcentaje de NaNs:', ascending=False) 
    
    return df_resumen_nans
    
#Indexar la columna del SpectrumType (LabelEncoding)
def encode_spectrum_type(df):
    #Se limpian los strings: quitando espacios y poniendolo en minuscula. 
    df['SpectrumType'] = df['SpectrumType'].str.strip().str.lower()
    
    #Se define un diccionario  para indexar los spectrum types en dicha columna
    espectros_indexados = {'powerlaw': 0, 'logparabola': 1, 'plsuperexpcutoff': 2 }
    
    #Mapea la columna con el diccionario de acuerdo a cada tipo de espectro
    df['SpectrumType'] = df['SpectrumType'].map(espectros_indexados)
    return df

#Normalizar los valores de los features:
def normalizar_features(df):
    #Se crea un objeto de StandardScaler que normaliza los datos (media 0, desviación estándar 1)
    obj_scaler = StandardScaler() #Usa Z-Score para normalizar
    
    #Se obtienen todas las columnas numéricas del DataFrame, filtra por tipo de dato y despues las convierte en lista
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    #Se elimina 'SpectrumType' de las columnas numéricas, porque no queremos normalizarla
    cols_numericas.remove('SpectrumType')  
    
    #Se normalizan solo las columnas numéricas seleccionadas
    df[cols_numericas] = obj_scaler.fit_transform(df[cols_numericas])
    return df
    

def count_infinities(df):
    # Verificar si hay valores infinitos (positivos o negativos)
    inf_counts = (df.isin([np.inf, -np.inf])).sum()
    
    # Devolver el conteo de infinitos por columna
    return inf_counts












