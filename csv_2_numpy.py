import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from datetime import datetime
# CSV einlesen
df = pd.read_csv("sensor_data.csv", delimiter=",", names=["Zeit", "Wert"], skiprows=1)

# Zeitspalte in Datetime umwandeln und neue Zeitspalten extrahieren
df["Zeit"] = pd.to_datetime(df["Zeit"])
df[["Jahr", "Monat", "Tag", "Stunde", "Minute"]] = df["Zeit"].apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute]).apply(pd.Series)

# Vollständige Zeitreihe erstellen
full_time_index = pd.date_range(df["Zeit"].min(), df["Zeit"].max(), freq="T")
df_full = pd.DataFrame({"Zeit": full_time_index})
df_full[["Jahr", "Monat", "Tag", "Stunde", "Minute"]] = df_full["Zeit"].apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute]).apply(pd.Series)

# Originalwerte einfügen und interpolieren
df = df_full.merge(df.drop(columns=["Zeit"]), on=["Jahr", "Monat", "Tag", "Stunde", "Minute"], how="left")
df["Wert"] = df["Wert"].interpolate(method="linear")

# Zeitspalte entfernen und Spalten neu anordnen
df = df.drop(columns=["Zeit"])[["Jahr", "Monat", "Tag", "Stunde", "Minute", "Wert"]]

jahr1 = int(df.iloc[900]['Jahr'])
monat1 = int(df.iloc[900]['Monat'])
tag1 = int(df.iloc[900]['Tag'])
stunde1 = int(df.iloc[900]['Stunde'])
minute1 = int(df.iloc[900]['Minute'])

jahr2 = int(df.iloc[-900]['Jahr'])
monat2 = int(df.iloc[-900]['Monat'])
tag2 = int(df.iloc[-900]['Tag'])
stunde2 = int(df.iloc[-900]['Stunde'])
minute2 = int(df.iloc[-900]['Minute'])

# Datum der ersten Zeile extrahieren
timestamp_first = pd.to_datetime(f"{jahr1}-{monat1}-{tag1} {stunde1}:{minute1}")
timestamp_900th_last = pd.to_datetime(f"{jahr2}-{monat2}-{tag2} {stunde2}:{minute2}")
#print(timestamp_first)
#print(timestamp_900th_last)
# Packe die Timestamps in ein Array
timestamps_array = np.array([timestamp_first, timestamp_900th_last])
#print(timestamps_array)
np.save("timestamps", timestamps_array)



# Letzte 900 Zeilen auswählen
#print(df.head(900))
#print(df_last_900.head())

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


array = df_to_numpy(df)
# Ausgabe des Numpy-Arrays
print(array.shape)
model = tf.keras.models.load_model(
    "fnn_v2.h5",
    custom_objects={"mse": MeanSquaredError()}
)
arr = []
input_datum = array[-900, :9].reshape(9,)
input_stand = array[-900:, 9].reshape(900,)
input = np.concatenate((input_datum, input_stand))#(909,)

for i in range(4): # funktioniert nur wenn input größer ist als output
    next_prediction = model.predict(input.reshape(1, -1), verbose=0).reshape(180, )
    arr = np.append(arr, next_prediction)

    input = input[9:]
    input = np.concatenate((input,next_prediction))
    input = input[180:]
    input = np.concatenate((array[-900+(i+1)*180, :9].reshape(9,),input)) #wegen der zeile wird das nicht unendlich autoregressiv laufen, dafür gehen die 9 datumswerte aus

#print(arr.shape)
#arr = np.concatenate([input_stand, arr ])
np.save("input", input_stand * 1000)
np.save("live_prediction", arr * 1000)

temp_arr = []
historic_prediction = []
historic_prediction_temp = []
historic_prediction_full = []
range_loop = 14 if datetime.now().hour < 12 else 15


cut_off_var = array.shape[0] - 12 * 60 * 14 - 900
print(cut_off_var)
#range_loop = 15
for i in range(range_loop):
    historic_datum = array[(12 * i) * 60 , :9].reshape(9,)
    historic_stand = array[(12 * i) * 60 : ((12 * i) + 15) * 60 , 9].reshape(900,) # refer to onenote skizze
    temp_arr = np.concatenate((historic_datum, historic_stand))

    historic_prediction = []
    for j in range(4):
        historic_prediction_temp = model.predict(temp_arr.reshape(1, -1), verbose=0).reshape(180, )
        historic_prediction = np.append(historic_prediction, historic_prediction_temp)


        temp_arr = temp_arr[9:]
        temp_arr = np.concatenate((temp_arr, historic_prediction_temp))
        temp_arr = temp_arr[180:]
        temp_arr = np.concatenate((array[(12 * i) * 60 + (j + 1) * 180, :9].reshape(9, ), temp_arr))

    historic_prediction_full = np.append(historic_prediction_full, historic_prediction)
print(len(historic_prediction_full))
historic_prediction_full = historic_prediction_full[:(len(historic_prediction_full)-(720-cut_off_var))]
#print(historic_prediction_full.shape)


print(array[15 * 60 : 15 * 60 + 12 * 60 * range_loop , 9].shape)
historic_input_full = array[15 * 60 : 15 * 60 + 12 * 60 * range_loop , 9]
#print(historic_input_full.shape)

fehler_array = np.abs(historic_prediction_full - historic_input_full)

import matplotlib.pyplot as plt

# Beispielarray mit 15120 Elementen (hier einfach eine Zahlenreihe von 0 bis 15119)
data = list(range(15120))

# Plot erstellen
plt.figure(figsize=(10, 4))
plt.plot(historic_prediction_full * 1000, linewidth=0.5)
plt.plot(historic_input_full * 1000, linewidth=0.5)
plt.plot(fehler_array * 1000, linewidth=0.5)
plt.plot(np.zeros(10080), linewidth=0.5)
plt.plot(np.full(10080, np.mean(fehler_array * 1000)), linewidth=0.5)
plt.plot(np.full(10080, np.max(fehler_array * 1000)), linewidth=0.5)
plt.title("Plot des Arrays")

plt.show()

np.save("historic_data", historic_input_full * 1000)
np.save("historic_predictions", historic_prediction_full * 1000)
np.save("error", fehler_array * 1000)
np.save("mean_error",np.full(10080, np.mean(fehler_array * 1000)) )
np.save("max_global_error",np.full(10080, np.max(fehler_array * 1000)) )


