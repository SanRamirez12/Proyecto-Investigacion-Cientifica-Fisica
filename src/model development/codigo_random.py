
for train_idx, val_idx in kfold.split(X_np, Y):
    print(f"\n=== Fold {fold_idx} ===")
    X_train, X_val = X_np[train_idx], X_np[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    #Arquitectura de red con un modelo
    model = Sequential([
        Dense(30, input_shape=(X_np.shape[1],), activation='relu', name='hidden_layer_1'),
        Dense(5, activation='softmax', name='output_layer')
    ])
    
    #Resumen del modelo
    model.summary()
    
    #Se compila el modelo
    #Optimizador: Adam, Perdida: sparse_categorical_crossentropy
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    #Se hace un Callback de con parada anticipada de 10 valores en caso de que el error no mejore
    #Tecnica de regularizacion para evitar overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    #Se entrena el modelo:
    print('\nEntrenando .......\n')
    
    #Se establece un tiempo inicial de entrenamiento
    t0 = time.time()
    
    #Se entrena el modelo con 500 epocas
    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        epochs=1000,
                        batch_size=64,
                        callbacks=[early_stop],
                        verbose=0)
    
    tf = time.time()
    print(f"Duración entrenamiento: {tf - t0:.2f} segundos")

    ##### Parte de Evaluacion ####################
    
    #Se evalua el test set de acuerdo con el modelo
    val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
    print(f"Precisión en validación: {val_acc:.4f}")

    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    #Se printea el reporte de classificacion de sklearn para ver las metricas y como funciona el modelo
    print("\nReporte de Clasificación:")
    print(classification_report(Y_val, y_pred, target_names=encoder.classes_))
    
    #Se incrementa el contador del fold en una unidad para continuar con el forloop:
    fold_idx += 1

# # #Printeamos la matriz de confusion:
# # umd.matriz_confusion(Y_test, y_pred, encoder.classes_)