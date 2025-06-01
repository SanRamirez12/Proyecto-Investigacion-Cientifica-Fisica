
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