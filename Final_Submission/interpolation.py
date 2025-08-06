import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib

def interpolation(d_alpha,h_c,r,w_t,l_t,w_o,dxIB,gamma):
    X_test=np.array([d_alpha,h_c,r,w_t,l_t,w_o,dxIB,gamma])
    scaler = joblib.load('interpolation_models/scaler.gz')
    X_test=scaler.transform([X_test])
    print(X_test.shape)
    columns=['T', 'TR', 'm_Cu', 'm_mag', 'cos_phi', 'VM', 'Temp']
    y_pred=[]
    
    model = keras.Sequential([
        keras.Input(shape=(X_test.shape[1],)),
        layers.Dense(512, activation=tf.keras.activations.gelu),
        layers.Dense(256, activation=tf.keras.activations.gelu),
        layers.Dense(128, activation=tf.keras.activations.gelu),
        layers.Dense(64, activation=tf.keras.activations.gelu),
        layers.Dense(32, activation=tf.keras.activations.gelu),
        layers.Dense(1),
    ])
    
    for i,col in enumerate(columns):
        model.load_weights(f"interpolation_models/best_model_{col}.keras")
        temp= model.predict(X_test, verbose=0).flatten()
        y_pred.append(temp[0])
    return y_pred

