import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

def create_train_data_generator(data, window_size, batch_size=100000):
    data = np.array(data, dtype=np.float16)  # Speicher sparen durch float32
    num_samples = data.shape[0] - window_size + 1  # Anzahl der Sliding Windows

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        # Sliding Window nur für den aktuellen Batch berechnen
        windows = np.lib.stride_tricks.sliding_window_view(data[start:end + window_size - 1, 12], window_size)

        # Batch von Features + Sliding Window zurückgeben (yield für iterativen Ansatz)
        for i in range(len(windows)):
            yield np.hstack((data[start + i, :12], windows[i]))


# Beispiel für Nutzung:
batch_size = 100000
window_size = 1500
data_clean_mit_zeit = []

for row in create_train_data_generator(d, window_size, batch_size):
    data_clean_mit_zeit.append(row)

# In NumPy-Array umwandeln
data_clean_mit_zeit = np.array(data_clean_mit_zeit, dtype=np.float16)
print(data_clean_mit_zeit.shape)  # Sollte eine speicherfreundliche Form haben

data = np.load("test_array.npy")
print(data.shape)
plt.plot(data[:,12]);plt.show()