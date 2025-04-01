from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import os


#definimos una funcion que plotea los histogramas dependiendo del parametro que le pasemos
def definirHistogramaFrecuencia_x_Parametro(path_archivo, n_feature, nombre_asignado):
    #Se asgina un feature del catalogo, solo para leer
    catalogo_agn = fits.open(name=path_archivo,mode="readonly")
    #Se accede al segundo elemento (índice 1) del archivo FITS
    datos = catalogo_agn[1].data
    #Se extrae la columna correspondiente al parámetro n_feature de la tabla de datos
    columna = datos[n_feature]
    #La columna extraída se convierte en un DataFrame de Pandas para facilitar su manipulación y análisis.
    columna_dataframe = pd.DataFrame(columna)
    #Se renombra la columna del DataFrame, cambiando el nombre predeterminado (0) por nombre_asignado.
    columna_dataframe.rename(columns={0: nombre_asignado}, inplace=True)
    
    #se realiza el ploteo del histograma de frecuencias porcentuales para el feature elegido:
    frecuencias = 100*columna_dataframe[nombre_asignado].value_counts()/len(columna)
    #Se utiliza el método plot de Pandas para crear un gráfico de barras que representa las frecuencias porcentuales de cada valor.
    plot = frecuencias.plot(kind='bar')
    #Se plotean los datos
    plt.xlabel(nombre_asignado)
    plt.ylabel('Frecuencia relativa (%)')
    plt.title('Histograma de Frecuencias Relativas Porcentuales')
    plt.show()

#Funcion que me lee el archivo actual 
def read_RawFile(nombre_archivo):
    #Se designa la ruta actual de este archivo:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta relativa al archivo .fits ubicado en ../data/raw/
    file_path = os.path.join(directorio_actual, '..', 'data', 'raw', nombre_archivo)
    return file_path
    
