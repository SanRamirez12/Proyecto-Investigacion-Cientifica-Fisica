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

#Convertimos infinitos a nans:
df_infs_a_nans = utde.inf_a_nan(df_spectype_encode)

# #Dataframe sobre los valores faltantes (NaNs): 
# df_resumen_nans = utde.estudio_nans(df_spectype_encode)
# print(df_resumen_nans)

# #Filas con Nans y su tipologia:
# df_filas_nans = utde.filas_con_nans(df_infs_a_nans)
# print(df_filas_nans)

#Borramos las 4 filas con nans 
df_limpio = utde.eliminar_filas_nans(df_infs_a_nans)

#Obtenemos un resumen de los datos que tenemos por clase y modelo espectral:
tabla_informativa = utde.resumen_fuentes_y_spectro(df_limpio)
# print(tabla_informativa)

#Normalizamos las columnas del dataframe para que entren bien en la ANN
df_norm = utde.normalizar_features(df_limpio)

print('###############################################################')
#Quedan 16 parametros: X con shape(7191,15) y Y los de CLASS1
df_actual = df_norm
print(df_actual) 



############## PLOTS ######################
# #Se plotea el heatmap con la matriz de correlacion bajo el metodo de Pearson
# utde.corr_matrix_heatmap(df_actual)

# #Se realiza el pairplot;
# utde.pairplot_features(df_actual)

############# Exp Archivos ##########################################
print('###############################################################')

utde.exportar_df_variantes(df_actual)
