import utils_model_dev as umd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import numpy as np
from sklearn.metrics import classification_report

#Empezamos leyendo los archivos. tomamos el datos sin fuentes no asociadas:
X, Y, encoder = umd.cargar_dataset('df_final_sin_UncAss.parquet', encoding='label', return_encoder=True)
# print(f'Features: {X} y target Values: {Y}')

#Dividir el dataset, se separa en train, val y test (60/20/20) con stratificación
#Dividimos 80% a Arrays temporales train/cv y 20% test
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42)
#Del 80% anterior, 25% para cv y 75% para training
X_train, X_cv, Y_train, Y_cv = train_test_split(
    X_temp, Y_temp, test_size=0.25, stratify=Y_temp, random_state=42)

#print(f'Training Set: {X_train}, CV-Set: {X_cv} y Test Set:{X_test}')

#Arquitectura de red con un modelo simple(sin regularizacion y solo un hidden layer):
model = Sequential([
    Dense(30, input_shape=(X.shape[1],), activation='relu', name='hidden_layer_1'),
    Dense(5, activation='softmax', name='output_layer')  
])
#Resumen del modelo
# model.summary()

#Se compila el modelo
#Optimizador: Adam, Perdida: sparse_categorical_crossentropy
model.compile(optimizer=Adam(learning_rate=0.001 ),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Se entrena el modelo:
print('\nEntrenando .......\n')
#Se establece un tiempo inicial de entrenamiento
t_inicio_entreno = time.time()
#Se hace un Callback de con parada anticipada de 10 valores en caso de que el error no mejore
#Tecnica de regularizacion para evitar overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

#Se entrena el modelo con 500 epocas
history = model.fit(X_train, Y_train,
                    validation_data=(X_cv, Y_cv),
                    epochs=500,
                    batch_size=64,
                    callbacks=[early_stop],
                    verbose=1)
#Se establece un tiempo final de entrenamiento
t_fin_entreno = time.time()
#Tiempo total:
diff_tiempo =t_fin_entreno - t_inicio_entreno
print(f'\nFin del entrenamiento. Duracion: {diff_tiempo:.2f}s\n')

#Printeamos las learning curves
umd.learning_curves(history)


##### Parte de Evaluacion ####################

#Se evalua el test set de acuerdo con el modelo
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'\nPrecisión del Test_set: {test_acc:.4f}\n')

#Se realiza un reporte de métricas
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

#Se printea el reporte de classificacion de sklearn para ver las metricas y como funciona el modelo
print("\n\nReporte de Clasificación:")
print(classification_report(Y_test, y_pred, target_names=encoder.classes_))

#Printeamos la matriz de confusion:
umd.matriz_confusion(Y_test, y_pred, encoder.classes_)


