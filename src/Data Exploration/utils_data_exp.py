#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from astropy.io import fits
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import time 


#Preseleccion de columnas (No incluye flujos por ahora.)
def select_columnas(headers):
    #Los nombres de columnas estan guardados en Ttypes del 1 al 79 dentro de headers, por tanto:
    ttypes = []
    for i in range(1, 80):  # Desde TTYPE1 hasta TTYPE79
        nombre_columna = f'TTYPE{i}'
        if nombre_columna in headers:
            ttypes.append(headers[nombre_columna])
    #print(ttypes) #Tipos de columnas
    '''
 A partir de los nombres de columnas, los vemos y elegimos de los 79 los que nos parecen:
Los uncertanties de las energias no los tomamos.
Las columnas 'Flux_Band', 'nuFnu_Band', 'Flux_History' son arrays de 8, 8 y 14 valores respectivamentes:
Las columnas 'Flux_Peak', 'Variability_Index', 'Frac_Variability' tienen nulls identificados como np.inf (5370,1,1 respectivamew)
De acuerdo con la explicacion fisica detras de la eliminacion de'Flux1000', 'Energy_Flux100' se debe a que
estas muestran flujos de energia dependientes de la intensidad y a la que se midieron y su distancia
y estos valores no son tan representativos como el espectro final y los indices de dichos espectros
por ende para este estudio mejor los eliminamos. 
Las columnas 'LP_EPeak' y 'PLEC_EPeak' tienen muchos NaNs y no nos sirven para el analisis
    '''
    #Lista de columnas seleccionadas:
    columnas_seleccionadas = [
        'SpectrumType', 'PL_Flux_Density', 'PL_Index', 'LP_Flux_Density', 
        'LP_Index', 'LP_beta', 'LP_SigCurv','PLEC_Flux_Density', 
        'PLEC_IndexS', 'PLEC_ExpfactorS', 'PLEC_Exp_Index', 'PLEC_SigCurv', 
        'Npred','Variability_Index', 'Frac_Variability', 'CLASS1'
        ]
    #catidad_select = len(columnas_seleccionadas)
    #Lista de columnas eliminadas:
    # columnas_eliminadas = [
    #     'Flux_Band', 'nuFnu_Band', 'Flux_History', 
    #     'Flux_Peak', 'LP_EPeak', 'PLEC_EPeak', 'Flux1000', 'Energy_Flux100'
    #     ]
    #cantidad_elim= len(columnas_eliminadas)
    
    #print(f"Seleccionamos una catidad de {catidad_select} columnas y eliminamos {cantidad_elim} columnas")
    return columnas_seleccionadas

#Leer el archivo y transformarlo como un data frame
def leer_fits(nombre_archivo):
    #Se designa la ruta actual de este archivo:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta relativa al archivo .fits ubicado en ../data/raw/
    file_path = os.path.join(directorio_actual, '..', '..', 'data', 'raw', nombre_archivo)

    #Se lee el archivo:
    with fits.open(name=file_path, mode="readonly") as datos_fits_4fgl:
        #print(datosFits_4FGL.info()) #Tipos de datos
        
        #Se accede al segundo elemento (índice 1) del archivo FITS: LAT_Point_Source_Catalog
        catalogo_fuentes = datos_fits_4fgl[1].data
        headers = datos_fits_4fgl[1].header #Visualizar info de headers
        
        #Se seleccionan las colmunas respectivas
        columnas_seleccionadas = select_columnas(headers)
        
        #Se crea un diccionario donde se filtran los datos por las columnas seleccionadas
        datos = {}
        for col in columnas_seleccionadas:
            #Se especifica que solo queremos datos de las cols seleccionadas
            col_data = catalogo_fuentes[col]
            #Se evita el error: Big-endian buffer not supported on little-endian compiler
            if col_data.dtype.byteorder == '>':  
                col_data = col_data.byteswap().newbyteorder()
            datos[col] = col_data
        
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
    
    # Se hace one-hot encoding manualmente con pandas
    dummies = pd.get_dummies(df['SpectrumType'], prefix='spectrum')
    
    #Se asegura que estén todas las columnas esperadas (por si falta alguna categoría en el dataset actual)
    for col in ['spectrum_powerlaw', 'spectrum_logparabola', 'spectrum_plsuperexpcutoff']:
        if col not in dummies.columns:
            dummies[col] = 0  # columna de ceros si no apareció esa categoría
    
    #Se convierten  booleanos a enteros explícitamente
    dummies = dummies.astype(int)
    
    #Se concatenan las columnas one-hot al dataframe original
    df =pd.concat([df.drop(columns=['SpectrumType']), dummies], axis=1)
    
    return df

#Metodo para contar infinitos en una columna del df
def contar_inf(df): #
    #Cuenta los valores infinitos por columna
    inf_counts = (df == np.inf).sum()
    neg_inf_counts = (df == -np.inf).sum()

    #Se obtiene el total de infinitos por columna
    total_inf = inf_counts + neg_inf_counts

    #Se filtran las columnas donde hay al menos un infinito
    total_inf = total_inf[total_inf > 0]

    #Devuelve un df
    resumen = pd.DataFrame({
        'Cantidad de infinitos': total_inf
    }).sort_values('Cantidad de infinitos', ascending=False)

    return resumen

#Aparecen como nulls en fluxpeak(5370),Variability_Index y Frac_Variability 
#(1y 1) dentro del fits pero pandas los detecta como infs
def inf_a_nan(df): 
    #Se toma la columna las otras dos del dataframe y se remplazan por nans
    cols_x_arreglar = ['Variability_Index', 'Frac_Variability']
    #Usamos el .replace para cambiar los infs por nans
    df[cols_x_arreglar] = df[cols_x_arreglar].replace([np.inf, -np.inf], np.nan)
    return df

#Elimina columnas que tienen un porcentaje de NaNs mayor al threshold.
def elimina_cols_alto_nans(df, threshold):
    #Se determina el porcetanje de nans de cada columna
    porcentaje_nans = df.isna().mean()
    
    #Encuentra las columnas que tienen más NaNs que el threshold, y las lista
    cols_a_eliminar = porcentaje_nans[porcentaje_nans > threshold].index.tolist()
    
    #Crea un nuevo dataframe que elimina las columnas listadas
    df = df.drop(columns=cols_a_eliminar)
    
    #Mensaje que me indique cuales se eliminaron
    print(f"Columnas eliminadas por alto porcentaje de NaNs (> {threshold*100}%): {cols_a_eliminar}")
    return df

#Metodo para ver los tipos de Nans por fila:
def filas_con_nans(df):

    #Selecionamos las filas con al menos un NaN y las ponemos en un df
    df_nans = df[df.isna().any(axis=1)].copy()
    
    #Se identifican las columnas con NaNs para cada fila (como lista)
    cols_con_nan = df_nans.isna().apply(lambda row: row[row].index.tolist(), axis=1)
    
    # Construir DataFrame resumen
    df_resumen = pd.DataFrame({
        'Columnas_con_NaNs': cols_con_nan,
        'SpectrumType': df_nans['SpectrumType'],
        'CLASS1': df_nans['CLASS1']
    }, index=df_nans.index)
    
    return df_resumen

#Elimnamos filas con nans(como son 4 no me importan mucho)
def eliminar_filas_nans(df):
    #Elimina todas las filas que contienen al menos un NaN en cualquier columna.
    df_limpio = df.dropna()
    return df_limpio

#Metodo para ver cantidad de fuentes por tipo y spectrumtype:
def resumen_fuentes_y_spectro(df):
    #Se mapean los valores numéricos a nombres legibles (si aún están como 0,1,2)
    tipo_spectro = {0: 'PowerLaw', 1: 'LogParabola', 2: 'PLSuperExpCutoff'}
    df['SpectrumTypeNombre'] = df['SpectrumType'].map(tipo_spectro)
    
    #Se obtiene una tabla conteo cruzado de pandas: SpectrumType vs CLASS1
    tabla_cruzada = pd.crosstab(df['CLASS1'], df['SpectrumTypeNombre'])

    #Se cuentan las fuentes por CLASS1
    conteo_total = df['CLASS1'].value_counts().sort_index()

    #Se inserta una columna con el total al inicio del DataFrame
    tabla_cruzada.insert(0, 'Total_fuentes', conteo_total)
    
    # Crear gráfico de barras apiladas
    tabla_cruzada.plot(kind='bar', stacked=False, figsize=(10, 6), colormap='viridis')
    
    # Estética del gráfico
    plt.title('Distribución de tipo de espectro por clase ')
    plt.xlabel('Clase (CLASS1)')
    plt.ylabel('Cantidad de fuentes')
    plt.xticks(rotation=45)
    plt.legend(title='SpectrumType')
    plt.tight_layout()
    plt.show()



#Normalizar los valores de los features: 
def normalizar_features(df):
    #generamos una copia
    df_copia = df.copy()
    #Se crea un objeto de StandardScaler que normaliza los datos (media 0, desviación estándar 1)
    obj_scaler = StandardScaler() #Usa Z-Score para normalizar
    
    #Se obtienen todas las columnas numéricas del DataFrame, filtra por tipo de dato y despues las convierte en lista
    cols_numericas = df_copia.select_dtypes(include=[np.number]).columns.tolist()
    
    #Se elimina 'SpectrumType' de las columnas numéricas, porque no queremos normalizarla
    for col_excluir in ['SpectrumType']:
        if col_excluir in cols_numericas:
            cols_numericas.remove(col_excluir) 
    
    #Se normalizan solo las columnas numéricas seleccionadas
    df_copia[cols_numericas] = obj_scaler.fit_transform(df_copia[cols_numericas])
    return df_copia

#Matriz de correlacion en un mapa calor:
def corr_matrix_heatmap(df):
    
    #Se copia el DataFrame para no modificar el original
    df_corr = df.copy()
    
    #Se eliminan columnas categóricas
    cols_excluidas = ['CLASS1', 'SpectrumType']
    df_corr = df_corr.drop(columns=[col for col in cols_excluidas if col in df_corr.columns])
        
    #Se crea una figura para guardar el heatmap
    plt.figure(figsize=(12,10))
    
    #Se crea la matriz de correlacion
    corr_matrix = df_corr.corr()
    
    #Mapa de calor para ver la matriz de correlacion
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".1f", linewidth=.5, 
                cbar=True, xticklabels='auto', yticklabels='auto', )
    #Se agrega un nombre
    plt.title('Mapa de calor de correlación de Pearson entre características')
    #Se muestra el ploteo
    plt.show()    
    
#Metodo para hacer el pairplot:
def pairplot_features(df):
    #Se mide el tiempo inicial del proceso:
    t_inicial = time.time()
    #Se copia el DataFrame para no modificar el original
    df_nuevo = df.copy()
    #Se eliminan columnas categóricas
    cols_excluidas = ['CLASS1']
    df_nuevo = df_nuevo.drop(columns=[col for col in cols_excluidas if col in df_nuevo.columns])
    #Selecciona los nombres de las columnas. 
    features = df_nuevo.columns.tolist()
    #Se elige con cual feature coloreamos el pairplot:
    label_col='CLASS1'
    #Se realiza el pairplot, coloreandolo con los label de CLASS1
    sns.pairplot(df[features + [label_col]], hue=label_col)

    #Se muestra el pairplot
    plt.show()
    #Se mide el tiempo final del proceso:
    t_final = time.time()
    
    #Se realiza la duracion y se muestra en consola
    duracion = t_final - t_inicial
    print(f"Tiempo de ejecución del pairplot: {duracion:.2f} segundos")

#Metodo para exportar el dataframe final como csv y parquet:
def exportar_df_variantes(df):
    #Ubicamos el punto donde queremos que se exporte en nuestro directorioÑ
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    carpeta_salida = os.path.join(directorio_actual, '..', '..', 'data', 'post preliminary analysis')
    
    #Se crea una carpeta de salida si no existe.
    os.makedirs(carpeta_salida, exist_ok=True)
    
    nombre_base='df_final'
    
    #Hacemos copias en .csv por si quiero verlas yo y .parquet par que sean menos pesados para el modelo.
    
    #Se realiza una copia fiel del DataFrame completo
    df.to_csv(f"{carpeta_salida}/{nombre_base}_completo.csv", index=False)
    df.to_parquet(f"{carpeta_salida}/{nombre_base}_completo.parquet", index=False)

    #Se exportan archivos solo con lasfilas con label 'UncAss' (fuentes no asosciadas) para el deployment
    df_uncass = df[df['CLASS1'] == 'UncAss']
    df_uncass.to_csv(f"{carpeta_salida}/{nombre_base}_solo_UncAss.csv", index=False)
    df_uncass.to_parquet(f"{carpeta_salida}/{nombre_base}_solo_UncAss.parquet", index=False)

    #Se exporta un archivo con todas las demás clases, excepto 'UncAss', para train, cv y test con el modelo
    df_no_uncass = df[df['CLASS1'] != 'UncAss']
    df_no_uncass.to_csv(f"{carpeta_salida}/{nombre_base}_sin_UncAss.csv", index=False)
    df_no_uncass.to_parquet(f"{carpeta_salida}/{nombre_base}_sin_UncAss.parquet", index=False)

    print("Exportación completada con éxito.")


