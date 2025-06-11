
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def arr2peak(prediction, historic):
    steps = prediction.shape[0] // 720
    pred_peak = []
    hist_peak = []
    p_trunc=[]
    h_trunc = []
    for i in range(steps):
        pred = prediction[i * 720: i * 720 + 720]
        hist = historic[i * 720: i * 720 + 720]
        p_trunc.append(pred)
        h_trunc.append(hist)
        hochwasser_pred = int(np.max(pred))
        hochwasser_real = int(np.max(hist))
        pred_peak.append(hochwasser_pred)
        hist_peak.append(hochwasser_real)
        #print('pred ', i, ': ', hochwasser_pred)
        #print('hist ', i, ': ', hochwasser_real)
    #p_trunc = prediction[:steps] #wir wollen nur wolle abschnitte von 720 werten, deshalb schneiden wir den rest am ende raus von den vorhersagen, die nicht 720 voll haben
    #h_trunc = historic[:steps]  #same
    return p_trunc, h_trunc, pred_peak, hist_peak, steps


def timestamp_generator(timestamp_first, steps, p, h):
    arr = []
    for i in range(steps):
        timestamp_first_plus_12 = timestamp_first + pd.Timedelta(hours=12)
        timestamps_array = np.array([timestamp_first, timestamp_first_plus_12, p[i], h[i]])
        timestamp_first = timestamp_first_plus_12
        arr.append(timestamps_array)

    return arr

p = np.load("historic_predictions.npy")
h = np.load("historic_data.npy")
timestamps = np.load("timestamps.npy", allow_pickle=True)
p_trunc, h_trunc, pred_peak, hist_peak, steps = arr2peak(p, h)
arr = timestamp_generator(timestamps[0], steps, pred_peak, hist_peak)
arr = np.array(arr)
p_trunc = np.array(p_trunc)
print(p_trunc.shape)


