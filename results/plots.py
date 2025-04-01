from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd


#definimos una funcion que plotea los histogramas dependiendo del parametro que le pasemos
def definirHistogramaFrecuencia_x_Parametro(nombre_archivo, n_feature, nombre_asignado):
    #Se asgina un feature del catalogo
    catalogo_agn = fits.open(name=nombre_archivo,mode="readonly")
    datos = catalogo_agn[1].data
    columna = datos[n_feature]
    columna_dataframe = pd.DataFrame(columna)
    columna_dataframe.rename(columns={0: nombre_asignado}, inplace=True)
    
    #se realiza el ploteo del histograma de frecuencias porcentuales para el feature elegido:
    frecuencias = 100*columna_dataframe[nombre_asignado].value_counts()/len(columna)
    plot = frecuencias.plot(kind='bar')
    plt.xlabel(nombre_asignado)
    plt.ylabel('Frecuencia relativa (%)')
    plt.title('Histograma de Frecuencias Relativas Porcentuales')
    plt.show()


# Archivo de latitudes altas |b|<=10 deg
nombre_archivo = "table-4LAC-DR2-h.fits" 
n_feature = 'CLASS'
nombre_asignado = 'Tipo_clase'
definirHistogramaFrecuencia_x_Parametro(nombre_archivo, n_feature, nombre_asignado)

####Archivo falta implementar algo que controle las frecuencias mas significativas tomando 
# en cuenta la anterior y la siguiente tipo metodo recursivo. 