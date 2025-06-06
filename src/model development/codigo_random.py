
# #Separamos el test set (15%) del training/validations sets (85%)
# X_trainval, X_test, y_trainval, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=47)
# print(f'\n Training/Validation sets shapes: X:{X_trainval.shape} y Y:{y_trainval.shape} \n Testing sets shapes: X:{X_test.shape} y Y:{Y_test.shape}')

# #Escalamos el test set para la evaluacion final:
# scaler = StandardScaler()
# X_test_escalado = X_test.copy()
# X_test_escalado[cols_a_escalar] = scaler.transform(X_test_escalado[cols_a_escalar])
# X_test_final = X_test_escalado.values.astype(np.float32)

# ####################### Evaluacion con test set ###################################################################
# #Se realiza la evaluación final en el conjunto de test (completamente independiente)
# y_test_pred_probs = model.predict(X_test_final)
# y_test_pred = np.argmax(y_test_pred_probs, axis=1)

# # Matriz de confusión final
# umd.matriz_confusion(Y_test, y_test_pred, labbels_clases=encoder.classes_)

# # Reporte final
# print("\n Reporte de Clasificación - Conjunto de Test")
# print(classification_report(Y_test, y_test_pred, target_names=encoder.classes_))


# #Separamos el test set (20%) del training/validations sets (80%)
# X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
# #Se escalan los datasets
# X_temp_final, X_test_final = umd.escalar_datasets(X_temp, X_test, cols_a_escalar)

# #Pesos por clase como hiperparámetros nombrados según el nombre real de la clase
# base_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
    
# #Genera los factores de ponderación como hiperparámetros nombrados con los nombres de clase reales
# class_weight_factors = {}
# for i, class_name in enumerate(encoder.classes_):
#     param_name = f'peso_de_{class_name}'
#     class_weight_factors[i] = trial.suggest_float(param_name, 0.1, 15.0)
    
# #Se multiplican los factores por los pesos base
# class_weights = {i: base_weights[i] * class_weight_factors[i] for i in range(len(encoder.classes_))}