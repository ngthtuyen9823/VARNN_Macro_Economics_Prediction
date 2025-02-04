# Data manipulation
import numpy as np

# Deep learning with TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

def create_var_predictions(data, model, lag_order, features):
    lagged_data = []
    for i in range(lag_order, len(data)):        
        pred = model.forecast(data[features].values[i-lag_order:i], steps=1)
        lagged_data.append(pred[0])
    
    return np.array(lagged_data)


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
