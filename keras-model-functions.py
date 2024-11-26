import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Función de pérdida Huber personalizada.
    """
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    small_error_loss = 0.5 * tf.square(error)
    big_error_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, small_error_loss, big_error_loss)

def create_sequences(data, time_steps):
    """
    Función para crear secuencias de datos de entrada y salida.
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def build_lstm_model(time_steps, input_size):
    """
    Función para construir el modelo LSTM mejorado.
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(time_steps, input_size)),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(50),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=huber_loss)
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
    """
    Función para entrenar el modelo LSTM mejorado.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    return history
