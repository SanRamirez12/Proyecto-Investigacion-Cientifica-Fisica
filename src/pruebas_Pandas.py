# -*- coding: utf-8 -*-

import utils_data_exp as utde


#Se lee el archivo:4FGL-DR4 V35 y se obtiene el DataFrame inicial.
dataFrame_inicial = utde.leerFITS('gll_psc_v35.fit') #No incluye flujos por ahora.
#print(dataFrame_inicial['CLASS1'])

#Se corrige el DataFrame inicial agregando nuevos lables: 
dataFrame_labeleado =utde.limpiar_labels_clases(dataFrame_inicial)
print(dataFrame_labeleado.describe())

#Holi

'''
Comandos para dataframes:
df.describe() :counts, mean, std, min, max,y %
df.info() : data tyoes, memoria, indices, columna

'''