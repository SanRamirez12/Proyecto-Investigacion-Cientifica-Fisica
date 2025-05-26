'''
    Optimizacion de hiperparametros con Optuna
'''
#Librerias de utils, y otras importantes:
import utils_model_dev as umd
import utils_hyperparameter_opt as uho
import time
import numpy as np
import optuna
import joblib

#Metodos de Skelearn
from sklearn.model_selection import train_test_split    
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

#Metodos Tensorflow
from tensorflow import keras
from tensorflow.keras import layers

#Metodos Optuna:
from optuna.integration import TFKerasPruningCallback
from optuna.visualization import plot_optimization_history, plot_param_importances

# ==================== CARGA Y ESCALADO ====================

X, Y, encoder = umd.cargar_dataset('df_final_sin_UncAss.parquet', encoding='label', return_encoder=True)

X = X.copy()
Y = Y.astype(np.int32)

X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
cols_no_escaladas = [col for col in X.columns if col.startswith("spectrum_")]
cols_a_escalar = [col for col in X.columns if col not in cols_no_escaladas]
X_temp_final, X_test_final = umd.escalar_datasets(X_temp, X_test, cols_a_escalar)

# ==================== SUBMUESTREO PARA OPTIMIZACIÓN ====================
X_sub, _, Y_sub, _ = train_test_split(X_temp_final, Y_temp, train_size=0.3, stratify=Y_temp, random_state=42)


# ==================== OPTUNA OBJECTIVE ====================
def objective(trial):
    # Hiperparámetros
    num_layers = trial.suggest_int('num_capas_ocultas', [1, 2, 3])
    num_units = trial.suggest_categorical('num_hidden_units', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', [0.0, 0.5, 0.8, 1.0])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'selu', 'elu', 'gelu'])
    optimizer_name = trial.suggest_categorical('optimizer', [
        'SGD', 'SGD_momentum', 'SGD_NAG', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'AdamW', 'Nadam'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 0.9) if 'momentum' in optimizer_name else 0.0
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Class weights (ponderaciones experimentales, ajustables)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_sub), y=Y_sub)
    class_weights = dict(enumerate(weights * trial.suggest_float('class_weight_factor', [0.5, 1.5, 2.0])))

    # Modelo
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_sub.shape[1],)))
    for _ in range(num_layers):
        model.add(layers.Dense(num_units, activation=activation))
        model.add(layers.Dropout(dropout_rate))


    model.add(layers.Dense(5, activation='softmax'))

    model.compile(
        optimizer=uho.get_optimizer(optimizer_name, learning_rate, momentum),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenamiento
    history = model.fit(
        X_sub, Y_sub,
        validation_split=0.2,
        epochs=100,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            TFKerasPruningCallback(trial, 'val_accuracy')
        ],
        class_weight=class_weights
    )

    # Evaluación
    y_pred = model.predict(X_sub)
    y_pred_labels = np.argmax(y_pred, axis=1)
    f1 = f1_score(Y_sub, y_pred_labels, average='weighted')
    return f1

# ==================== EJECUCIÓN DEL ESTUDIO ====================
study = optuna.create_study(
    direction='maximize', 
    study_name='AGN_opt', 
    sampler=optuna.samplers.TPESampler(), 
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

study.optimize(objective, n_trials=1000)

# ==================== VISUALIZACIÓN Y GUARDADO ====================
joblib.dump(study, 'optuna_agn_study.pkl')
plot_optimization_history(study).show()
plot_param_importances(study).show()

print("Best trial:")
trial = study.best_trial
print(f"  Value (Weighted F1): {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
