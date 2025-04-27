


def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
    import tensorflow as tf
    from tensorflow.keras import backend as k
    from tensorflow.keras.models import Sequential
    
    k.clear_session()   
    rede_neural = Sequential([
    tf.keras.layers.InputLayer(shape=(30,)),
    tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer= kernel_initializer ),
    tf.keras.layers.Dropout(rate= 0.2),   
    tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer= kernel_initializer ),
    tf.keras.layers.Dropout(rate= 0.2),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
    rede_neural.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return rede_neural
