import utils_data_exp as utde


#Se lee el archivo:4FGL-DR4 V35 y se obtiene el DataFrame inicial.
df_inicial = utde.leer_fits('gll_psc_v35.fit') #No incluye flujos por ahora.
#print(dataFrame_inicial['CLASS1'])

#Se corrige el DataFrame inicial agregando nuevos lables: 
df_labeleado =utde.limpiar_labels_clases(df_inicial)
#print(dataFrame_labeleado)

#Indexamos el spectrum type de las fuentes:
df_spectype_encode = utde.encode_spectrum_type(df_labeleado)
#print(df_spectype_encode)

#Corregimos los infinitos por valores Nan en el dataframe
df_infs_a_nans = utde.inf_a_nan(df_spectype_encode)
# print(df_infs_a_nans)


#Dataframe sobre los valores faltantes (NaNs): 
#df_resumen_nans = utde.estudio_nans(df_infs_a_nans)
#print(df_resumen_nans)
#Determinamos por ahora que queremos quitar los rows con features con muchos nans (Los Peaks)
#Por ahora ignoramos la imputacion de datos.

#Decidimos elimnar a las columnas con un threshold mayor al 10% (Basicamente los 3 peaks)
df_sin_nans = utde.elimina_cols_alto_nans(df_infs_a_nans, 0.1)


df_actual = df_sin_nans
print(df_actual) #Quedan 18 parametros

#Se plotea el heatmap con la matriz de correlacion bajo el metodo de Pearson
#utde.corr_matrix_heatmap(df_actual)

#Se realiza el pairplot;
#utde.pairplot_features(df_actual)
