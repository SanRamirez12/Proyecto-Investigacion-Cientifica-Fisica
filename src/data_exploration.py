import os
import utils_data_exp as utde


# Archivo de latitudes altas |b|<=10 deg:
nombre_archivo = 'table-4LAC-DR2-h.fits'
n_feature = 'CLASS' #Nombre de la columna a visualizar
nombre_asignado = 'Tipo_clase' #Nombre que le ponemos
#Me lee el archivo raw de la carpeta de datos raw
path_archivo = utde.read_RawFile(nombre_archivo) 
#Se plotean los datos del archivo fits
utde.definirHistogramaFrecuencia_x_Parametro(path_archivo, n_feature, nombre_asignado)

####Archivo falta implementar algo que controle las frecuencias mas significativas tomando 
# en cuenta la anterior y la siguiente tipo metodo recursivo. 