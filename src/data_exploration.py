import utils_data_exp as utde


#Se lee el archivo:4FGL-DR4 V35 y se obtiene el DataFrame inicial.
df_inicial = utde.leer_fits('gll_psc_v35.fit') #No incluye flujos por ahora.
#print(dataFrame_inicial['CLASS1'])

#Se corrige el DataFrame inicial agregando nuevos lables: 
df_labeleado =utde.limpiar_labels_clases(df_inicial)
#print(dataFrame_labeleado)

#Dataframe sobre los valores faltantes (NaNs): 
#df_resumen_nans = utde.estudio_NaNs(df_labeleado)
#print(df_resumen_nans)

#Indexamos el spectrum type de las fuentes:
df_spectype_encode = utde.encode_spectrum_type(df_labeleado)
print(df_spectype_encode)


#Contar infinitos:
infinitos = utde.count_infinities(df_spectype_encode)
print(infinitos)

# df_normalizado = utde.normalizar_features(df_spectype_encode)
# print(df_normalizado)