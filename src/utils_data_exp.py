#import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.io import fits

#Preseleccion de columnas (No incluye flujos por ahora.)
def select_Columnas(headers):
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
def leerFITS(nombre_archivo):
    #Se designa la ruta actual de este archivo:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta relativa al archivo .fits ubicado en ../data/raw/
    file_path = os.path.join(directorio_actual, '..', 'data', 'raw', nombre_archivo)

    #Se asgina un feature del catalogo, solo para leer
    datosFits_4FGL = fits.open(name=file_path, mode="readonly")
    #print(datosFits_4FGL.info()) #Tipos de datos
    
    #Se accede al segundo elemento (índice 1) del archivo FITS: LAT_Point_Source_Catalog
    catalogo_fuentes = datosFits_4FGL[1].data
    
    #Visualizar info de headers:
    headers = datosFits_4FGL[1].header
    
    #Se seleccionan las colmunas respectivas
    columnas_seleccionadas = select_Columnas(headers)
    
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

