#Metodos Tensorflow
from tensorflow.keras import optimizers


#Funcion para obtener optimizador:
def get_optimizer(name, lr, momentum):
    return {
        'SGD': optimizers.SGD(learning_rate=lr),
        'SGD_momentum': optimizers.SGD(learning_rate=lr, momentum=momentum),
        'SGD_NAG': optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True),
        'RMSprop': optimizers.RMSprop(learning_rate=lr),
        'Adagrad': optimizers.Adagrad(learning_rate=lr),
        'Adadelta': optimizers.Adadelta(learning_rate=lr),
        'Adam': optimizers.Adam(learning_rate=lr),
        'AdamW': optimizers.AdamW(learning_rate=lr),
        'Nadam': optimizers.Nadam(learning_rate=lr)
    }[name]
