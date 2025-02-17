# Data manipulation
import numpy as np

# Deep learning with TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers



import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

@tf.keras.utils.register_keras_serializable()
def custom_output_layer(x, var_weights, var_bias):
    return tf.matmul(x, var_weights) + var_bias

def build_varnn2(input_dim, num_variables, learning_rate, var_weights, var_bias):
    """
    Builds the VARNN model according to the paper:
      - Input: (input_dim, num_variables)
      - Flatten: converts the input to a vector
      - Dense layer with hidden_units = input_dim*num_variables and sigmoid activation (nonlinear processing)
      - Output layer: applies a linear transformation using var_weights and var_bias

    Parameters:
      - input_dim: number of lags (best_lags)
      - num_variables: number of variables in the time series
      - learning_rate: learning rate for the optimizer.
      - var_weights: weight matrix from VAR, expected shape (input_dim*num_variables, num_variables).
      - var_bias: bias vector from VAR, shape (num_variables,).

    Returns:
      - model: the compiled VARNN model.
    """
    hidden_units = input_dim * num_variables  # Tính tự động số đơn vị ẩn

    # Chuyển trọng số và bias thành tensor
    var_weights = tf.constant(var_weights, dtype=tf.float32)
    var_bias = tf.constant(var_bias, dtype=tf.float32)
    # Kiểm tra kích thước: var_weights.shape[0] phải bằng hidden_units
    assert var_weights.shape[0] == hidden_units, (
        f"Dimension mismatch: var_weights should have {hidden_units} rows, but got {var_weights.shape[0]}."
    )
    
    # Định nghĩa đầu vào có shape (input_dim, num_variables)
    inputs = tf.keras.Input(shape=(input_dim, num_variables))
    # Flatten đầu vào thành vector 1 chiều (batch_size, input_dim*num_variables)
    x = layers.Flatten()(inputs)
    # Dense layer với hidden_units và activation sigmoid
    hidden = layers.Dense(hidden_units, activation='sigmoid')(x)
    
    # Sử dụng Lambda layer với hàm custom đã đăng ký
    outputs = layers.Lambda(
        custom_output_layer,
        output_shape=(num_variables,),
        arguments={'var_weights': var_weights, 'var_bias': var_bias}
    )(hidden)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

def create_var_predictions1(data, model, lag_order, features):
    lagged_data = []
    for i in range(lag_order, len(data)):
        # Dự báo lag_order bước thay vì 1 bước
        pred = model.forecast(data[features].values[i-lag_order:i], steps=lag_order)
        lagged_data.append(pred)
    
    return np.array(lagged_data)


def create_var_predictions(data, model, lag_order, features):
    lagged_data = []
    for i in range(lag_order, len(data)):        
        pred = model.forecast(data[features].values[i-lag_order:i], steps=1)
        lagged_data.append(pred[0])
    
    return np.array(lagged_data)


import tensorflow as tf

def build_ffnn(input_dim, output_dim, hidden_layer_sizes, learning_rate):
    """
    Xây dựng mô hình FFNN đơn giản với TensorFlow Keras.
    
    Parameters:
        input_dim (int): Số lượng đặc trưng đầu vào.
        output_dim (int): Số lượng đầu ra.
        hidden_layer_sizes (int hoặc list): Số nơ-ron của mỗi lớp ẩn.
            Nếu truyền vào một số nguyên, sẽ hiểu là có 1 lớp ẩn với số nơ-ron tương ứng.
        learning_rate (float): Tốc độ học dùng cho optimizer.
        
    Returns:
        model: Một instance của tf.keras.models.Model đã được biên dịch.
    """
    # Nếu hidden_layer_sizes là một số nguyên, chuyển nó thành danh sách
    if isinstance(hidden_layer_sizes, int):
        hidden_layer_sizes = [hidden_layer_sizes]
    
    # Khởi tạo mô hình Sequential
    model = tf.keras.models.Sequential()
    
    # Thêm lớp input (hoặc có thể kết hợp trực tiếp vào lớp Dense đầu tiên)
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    
    # Thêm các lớp ẩn với activation là ReLU
    for units in hidden_layer_sizes:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    
    # Lớp đầu ra không có activation (hoặc có thể thêm activation phù hợp nếu cần)
    model.add(tf.keras.layers.Dense(output_dim))
    
    # Biên dịch mô hình sử dụng optimizer Adam và loss Mean Squared Error (MSE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def build_varnn(input_dim, output_dim, learning_rate, hidden_layer_sizes):
    model = Sequential([Input(shape=(input_dim,))])
    
    if isinstance(hidden_layer_sizes, int):
        hidden_layer_sizes = [hidden_layer_sizes]

    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
        model.add(Dropout(0.2))
    
    model.add(Dense(output_dim))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='mse', 
                  metrics=['mae'])
    return model
