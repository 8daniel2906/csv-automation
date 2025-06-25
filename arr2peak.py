
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
def warnung_generator(upper_peak, hist_peak):
    schwelle = 750 #das ist PI-model abhängig und kmeans abhängig
    hist_peak = np.array(hist_peak)
    upper_peak = np.array(upper_peak)
    mask11 = hist_peak >= schwelle      # Ground Truth: Überschwemmung
    mask12 = hist_peak < schwelle       # Ground Truth: keine Überschwemmung

    mask21 = upper_peak >= schwelle     # Prediction: Warnung
    mask22 = upper_peak < schwelle      # Prediction: keine Warnung

        # TP, FN, FP, TN
    TP = np.sum(mask11 & mask21)
    FN = np.sum(mask11 & mask22)
    FP = np.sum(mask12 & mask21)
    TN = np.sum(mask12 & mask22)

    print(f"Warn.und HW.: {TP}, Entwarn. aber HW: {FN}, Warn. aber K.HW: {FP}, Entwarn. und K. HW: {TN} ")



p = np.load("upper_historic.npy")
h = np.load("historic_predictions.npy")
timestamps = np.load("timestamps.npy", allow_pickle=True)
u_trunc, h_trunc, upper_peak, hist_peak, steps = arr2peak(p, h)
arr = timestamp_generator(timestamps[0], steps, upper_peak, hist_peak)
arr = np.array(arr)

warnung_generator(upper_peak, hist_peak)


