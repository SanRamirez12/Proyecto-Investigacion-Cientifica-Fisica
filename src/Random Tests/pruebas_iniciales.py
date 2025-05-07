from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd

#Seleccionamos el archivo .fit que queremos abrir
nombre_archivo = "gll_psc_v35.fit" 
#Se lee el archivo ya que no queremos modifarlo como tal
catalogo_fuentes = fits.open(name=nombre_archivo,mode="readonly")

#despliega la informacion del fit file de forma general
#catalogo_fuentes.info()

#Nos lee el LAT_Point_Source_Catalog del archivo fits,
# en este caso las 7195 fuentes con 79 parametros
datos = catalogo_fuentes[1].data
catalogo_fuentes.close() #Cerramos despues de trabjar con el archivo para ahorrar memoria
# print(datos[3][69]) # Columna 69 del row 3 (fuente 4 y header type 70)

#Seleccionamos una columna esxpecifica dentro del point source
clase1 = datos['CLASS1']
dec = datos['DEJ2000 ']

#Se crea un array con la clasificacion de las fuentes dl plano galactico
clase1_filtrado =[]
for i in range(len(dec)):
   if abs(dec[i])<=10:
       clase1_filtrado.append(clase1[i])

# Se crea un DataFrame que lee el array filtrado (el DataFrame es un objeto de tipo narray mx1 con m=7195)
fuentes_clase_df = pd.DataFrame(clase1_filtrado)
# Se renombra la columna del DataFrame
fuentes_clase_df.rename(columns={0: 'Clase1'}, inplace=True)

# Supuestamente rellena o elimina null rows
# fuentes_clase_df.fillna('Nulo', inplace=True)
# fuentes_clase_df.dropna(inplace=True)

# Se realiza un histograma de frecuencias relativas porcentuales
frecuencias = 100*fuentes_clase_df['Clase1'].value_counts()/len(clase1_filtrado)
plot = frecuencias.plot(kind='bar')
plt.xlabel('Categorías de clases de objetos')
plt.ylabel('Frecuencia relativa (%)')
plt.title('Histograma de Frecuencias Relativas Porcentuales')
plt.show()

