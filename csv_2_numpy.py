import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from datetime import datetime
import joblib
import sklearn
print(sklearn.__version__)#1.7.0

#######################################################################
def assign_clusters_inference(prediction, chunk_size, trained_kmeans):
    chunks = []
    chunk_ranges = []
    for start in range(0, len(prediction), chunk_size):
        end = start + chunk_size
        if end <= len(prediction):
            chunks.append(prediction[start:end])
            chunk_ranges.append((start, end))
    chunks = np.array(chunks)
    predicted_labels = trained_kmeans.predict(chunks)
    cluster_labels = np.full(len(prediction), np.nan)
    for i, (start, end) in enumerate(chunk_ranges):
        cluster_labels[start:end] = predicted_labels[i]
    return cluster_labels



def assign_time_bins_full_array(array_len, n_bins, packet_size):
    import numpy as np

    assert array_len % packet_size == 0, f"Länge muss ein Vielfaches von {packet_size} sein!"

    full_time_bin_array = []

    for packet_start in range(0, array_len, packet_size):
        time_bins = np.linspace(0, packet_size, n_bins + 1)
        # Lokale Indizes innerhalb des Pakets (0–719)
        local_indices = np.arange(packet_size)
        local_bins = np.digitize(local_indices, time_bins, right=False) - 1
        full_time_bin_array.extend(local_bins)

    return np.array(full_time_bin_array)

def get_quantile_bounds_from_labels(cluster_labels, time_labels, interval_matrix):
    assert len(cluster_labels) == len(time_labels), "Label-Arrays müssen gleich lang sein."

    lo_array = np.full(len(cluster_labels), np.nan)
    hi_array = np.full(len(cluster_labels), np.nan)

    for i in range(len(cluster_labels)):
        c = int(cluster_labels[i])
        t = int(time_labels[i])
        if not (np.isnan(c) or np.isnan(t)):
            lo, hi = interval_matrix[c][t]
            lo_array[i] = lo
            hi_array[i] = hi

    return lo_array, hi_array
# Laden
model_bundle = joblib.load('kmeans_interval_model8.pkl')
# Zugriff auf Inhalte
kmeans = model_bundle['kmeans']
interval_matrix = model_bundle['interval_matrix']
prediction1 = model_bundle['prediction1']

chunk = 180
cluster = 10
time_teile = 10

time_labels_klein = assign_time_bins_full_array(720, time_teile,720)

###################################################################################


#die jetzige version kommt noch nicht gut klar mit sensordatenlücken, die über eine Stunde hinausgehen
#umwandlung in ein df mit 2 spalten
df = pd.read_csv("sensor_data.csv", delimiter=",", names=["Zeit", "Wert"], skiprows=1)
#sensorfehler enthalten messungen = 0. die werden entfernt
df = df[df["Wert"] != 0]
#Zeit-spalte kriegt datentyp passend zu einem datum
df["Zeit"] = pd.to_datetime(df["Zeit"])
#sensor misst nur alle paar minuten, in nicht gemessenen minuten gibt es auch keinen wasserstand.
#man muss die zeitangaben mit lücken dann mit den fehlenden zeitangaben auffüllen und die wasserstände
#mit NaN ersetzen --> also mehr zeilen adden
df_full = pd.DataFrame({"Zeit": pd.date_range(df["Zeit"].min(), df["Zeit"].max(), freq="min")})
#left join und dann die NaN Werte interpolieren
df = df_full.merge(df, on="Zeit", how="left").interpolate(method="linear")


df[["Jahr", "Monat", "Tag", "Stunde", "Minute"]] = df["Zeit"].apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute]).apply(pd.Series)
df = df.drop(columns=["Zeit"])[["Jahr", "Monat", "Tag", "Stunde", "Minute", "Wert"]]

#man macht hier 840 ,weil hier geplottet wird, nicht als input für das modell verwendet wird
jahr1 = int(df.iloc[840]['Jahr'])
monat1 = int(df.iloc[840]['Monat'])
tag1 = int(df.iloc[840]['Tag'])
stunde1 = int(df.iloc[840]['Stunde'])
minute1 = int(df.iloc[840]['Minute'])

jahr2 = int(df.iloc[-840]['Jahr'])
monat2 = int(df.iloc[-840]['Monat'])
tag2 = int(df.iloc[-840]['Tag'])
stunde2 = int(df.iloc[-840]['Stunde'])
minute2 = int(df.iloc[-840]['Minute'])


timestamp_first = pd.to_datetime(f"{jahr1}-{monat1}-{tag1} {stunde1}:{minute1}")
timestamp_840th_last = pd.to_datetime(f"{jahr2}-{monat2}-{tag2} {stunde2}:{minute2}")
timestamps_array = np.array([timestamp_first, timestamp_840th_last])

np.save("timestamps", timestamps_array) #wird in app.py verwendet zum plotten

#hier machen wir features rein: zyklisches embedding der zeitangaben sowie normalisierung der wasserstandswerte

def df_to_numpy(df):
    # Jahr - 200 und durch 1000 teilen
    df['Jahr'] = (df['Jahr'] - 2000) / 1000.0

    # Monat zyklisch in Sinus und Kosinus umwandeln
    df['Monat_sin'] = np.sin(2 * np.pi * df['Monat'] / 12)
    df['Monat_cos'] = np.cos(2 * np.pi * df['Monat'] / 12)

    # Tag zyklisch in Sinus und Kosinus umwandeln
    df['Tag_sin'] = np.sin(2 * np.pi * df['Tag'] / 31)
    df['Tag_cos'] = np.cos(2 * np.pi * df['Tag'] / 31)

    # Stunde zyklisch in Sinus und Kosinus umwandeln
    df['Stunde_sin'] = np.sin(2 * np.pi * df['Stunde'] / 24)
    df['Stunde_cos'] = np.cos(2 * np.pi * df['Stunde'] / 24)

    # Minute zyklisch in Sinus und Kosinus umwandeln
    df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
    df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)

    # Wert durch 1000 teilen
    df['Wert'] = df['Wert'] / 1000.0

    # Die relevanten Spalten in der gewünschten Reihenfolge
    df_transformed = df[['Jahr', 'Monat_sin', 'Monat_cos', 'Tag_sin', 'Tag_cos',
                         'Stunde_sin', 'Stunde_cos', 'Minute_sin', 'Minute_cos', 'Wert']]

    # Numpy-Array erstellen
    return df_transformed.to_numpy()

array = df_to_numpy(df) #data transformation



df = pd.DataFrame(array)
df.columns = ["col1", "col2", "col3", "col4", "col5", "col6","col7","col8","col9","water_level"]
# mehr features , diesmal moving averages #minutes_per_week und minutes_per_month sind hier aber erstmal gleich inhaltlich ,weil ich nur daten der letzen woche entnehme
minutes_per_day = 60 * 24 #tägliche Durchschnitte
minutes_per_week = minutes_per_day * 7 #wöchentliche durchschnitte
minutes_per_month = minutes_per_day * 30 #monatliche Durchschnitte !!!! das feature macht eigentlich noch keinen sinn , wenn ich nur für eine Woche  die Vorhersagendemonstration mache
# Berechnung der Moving Averages
df["MA_Day"] = df["water_level"].rolling(window=minutes_per_day, min_periods=1).mean()
df["MA_Week"] = df["water_level"].rolling(window=minutes_per_week, min_periods=1).mean()
df["MA_Month"] = df["water_level"].rolling(window=minutes_per_month, min_periods=1).mean()
# Spalten neu anordnen (Moving Averages an den Anfang)
df = df[["MA_Month", "MA_Week", "MA_Day"] + df.columns[:-3].tolist()]

array = df.to_numpy()
np.save("test_array",array)
print("array:  ",array.shape )

#modell des kaggle notebooks wird geladen
model = tf.keras.models.load_model(
    "fnn.h5",
    custom_objects={"mse": MeanSquaredError()}
)

arr = []
input_datum = array[-840, :12].reshape(12,)
input_stand = array[-840:, 12].reshape(840,)
input = np.concatenate((input_datum, input_stand))
#vorhersage für den ersten plot wird erstellt , live-vorhersage:
for i in range(2): # funktioniert nur wenn input größer ist als output
    next_prediction = model.predict(input.reshape(1, -1), verbose=0).reshape(360, )
    arr = np.append(arr, next_prediction)
    input = input[12:]
    input = np.concatenate((input,next_prediction))
    input = input[360:]
    input = np.concatenate((array[-840+(i+1)*360, :12].reshape(12,),input)) #wegen der zeile wird das nicht unendlich autoregressiv laufen, dafür gehen die 9 datumswerte aus

cluster_labels2 = assign_clusters_inference(arr*100, chunk, kmeans)
q_low, q_up = get_quantile_bounds_from_labels(cluster_labels2, time_labels_klein, interval_matrix)
lower = arr*1000 - np.abs(q_low)
upper = arr*1000 + np.abs(q_up)


np.save("input", input_stand * 1000) #für app.py
np.save("live_prediction", arr * 1000) #für app.py
np.save("lower",lower)
np.save("upper",upper)
#vorhersage für den zweiten plot, wöchentliche vorhersage:

temp_arr = []
historic_prediction = []
historic_prediction_temp = []
historic_prediction_full = []
lower_hist_ = []
upper_hist_ = []
range_loop = 15 if datetime.now().hour < 12 else 16     # range_loop = 15 if datetime.now().hour < 12 else 16


cut_off_var = array.shape[0] - 12 * 60 * (range_loop-1) - 840
print(cut_off_var)

for i in range(range_loop):
    historic_datum = array[(12 * i) * 60 , :12].reshape(12,)
    historic_stand = array[(12 * i) * 60 : ((12 * i) + 14) * 60 , 12].reshape(840,) # refer to onenote skizze
    temp_arr = np.concatenate((historic_datum, historic_stand))
    #print(historic_datum.shape)
    #print(historic_stand.shape)


    historic_prediction = []
    for j in range(2): #autoregressiv
        historic_prediction_temp = model.predict(temp_arr.reshape(1, -1), verbose=0).reshape(360, )

        historic_prediction = np.append(historic_prediction, historic_prediction_temp)



        temp_arr = temp_arr[12:]
        temp_arr = np.concatenate((temp_arr, historic_prediction_temp))
        temp_arr = temp_arr[360:]
        temp_arr = np.concatenate((array[(12 * i) * 60 + (j + 1) * 360, :12].reshape(12, ), temp_arr))
        #print(temp_arr.shape)

    cluster_labels2 = assign_clusters_inference(1000*historic_prediction, chunk, kmeans)  # chunk, kmeans global
    q_low_hist, q_up_hist = get_quantile_bounds_from_labels(cluster_labels2, time_labels_klein, interval_matrix)
    lower_hist = historic_prediction * 1000 - np.abs(q_low_hist)
    upper_hist = historic_prediction * 1000 + np.abs(q_up_hist)
    lower_hist_.append(lower_hist)
    upper_hist_.append(upper_hist)
    historic_prediction_full = np.append(historic_prediction_full, historic_prediction)

lower_hist_ = np.array(lower_hist_)
upper_hist_ =  np.array(upper_hist_)
lower_hist_ = lower_hist_.flatten()
upper_hist_ = upper_hist_.flatten()

historic_prediction_full = historic_prediction_full[:(len(historic_prediction_full)-(720-cut_off_var))]
lower_hist_ = lower_hist_[:(len(lower_hist_)-(720-cut_off_var))]
upper_hist_ = upper_hist_[:(len(upper_hist_)-(720-cut_off_var))]

historic_input_full = array[14 * 60 : 14 * 60 + 12 * 60 * range_loop , 12]

fehler_array = np.abs(historic_prediction_full - historic_input_full)


#nur zum test lokal
import matplotlib.pyplot as plt

# Beispielarray mit 15120 Elementen (hier einfach eine Zahlenreihe von 0 bis 15119)
data = list(range(15120))
plt.figure(figsize=(10, 4))
plt.plot(lower_hist_ , linewidth=0.5)
plt.plot(upper_hist_ , linewidth=0.5)
plt.plot(historic_prediction_full * 1000, linewidth=0.5)
plt.plot(historic_input_full * 1000, linewidth=0.5)
plt.plot(fehler_array * 1000, linewidth=0.5)
plt.plot(np.zeros(10080), linewidth=0.5)
plt.plot(np.full(10080, np.mean(fehler_array * 1000)), linewidth=0.5)
plt.plot(np.full(10080, np.max(fehler_array * 1000)), linewidth=0.5)
plt.title("Plot des Arrays")



plt.show()

#ergebnisse speichern
np.save("historic_data", historic_input_full * 1000)#für app.py
np.save("historic_predictions", historic_prediction_full * 1000)#für app.py
np.save("error", fehler_array * 1000)#für app.py
np.save("mean_error",np.full(10080, np.mean(fehler_array * 1000)) )#für app.py
np.save("max_global_error",np.full(10080, np.max(fehler_array * 1000)) )#für app.py
np.save("lower_historic", lower_hist_)
np.save("upper_historic", upper_hist_)
np.save("test_arr", array )
