
import numpy as np
def arr2peak(prediction, historic):
    steps = prediction.shape[0] // 720
    pred_peak = []
    hist_peak = []
    for i in range(steps):
        pred = prediction[i * 720: i * 720 + 720]
        hist = historic[i * 720: i * 720 + 720]

        hochwasser_pred = int(np.max(pred))
        hochwasser_real = int(np.max(hist))
        pred_peak.append(hochwasser_pred)
        hist_peak.append(hochwasser_real)
        #print('pred ', i, ': ', hochwasser_pred)
        #print('hist ', i, ': ', hochwasser_real)
    return pred_peak, hist_peak
p = np.load("historic_predictions.npy")
h = np.load("historic_data.npy")
timestamps = np.load("timestamps.npy", allow_pickle=True)
print(timestamps)
p, h = arr2peak(p, h)

