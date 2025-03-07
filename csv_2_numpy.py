import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from datetime import datetime

#die jetzige version kommt noch nicht gut klar mit sensordatenlücken, die über eine Stunde hinausgehen

df = pd.read_csv("sensor_data.csv", delimiter=",", names=["Zeit", "Wert"], skiprows=1)
df = df[df["Wert"] != 0]
df["Zeit"] = pd.to_datetime(df["Zeit"])
df[["Jahr", "Monat", "Tag", "Stunde", "Minute"]] = df["Zeit"].apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute]).apply(pd.Series)
full_time_index = pd.date_range(df["Zeit"].min(), df["Zeit"].max(), freq="T")
df_full = pd.DataFrame({"Zeit": full_time_index})
df_full[["Jahr", "Monat", "Tag", "Stunde", "Minute"]] = df_full["Zeit"].apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute]).apply(pd.Series)
df = df_full.merge(df.drop(columns=["Zeit"]), on=["Jahr", "Monat", "Tag", "Stunde", "Minute"], how="left")
df["Wert"] = df["Wert"].interpolate(method="linear")#falls werte einfach aus dem sensor fehlen, also man pro minute nicht einen wert hat (passiert in den 2 aktuellsten tagen immer sowieso)
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

df = pd.DataFrame(array)  # your_data = dein Numpy-Array oder DataFrame
df.columns = ["col1", "col2", "col3", "col4", "col5", "col6","col7","col8","col9","water_level"]
# mehr features , diesmal moving averages
minutes_per_day = 60 * 24 #tägliche Durchschnitte
minutes_per_week = minutes_per_day * 7 #wöchentliche durchschnitte
minutes_per_month = minutes_per_day * 30 #monatliche Durchschnitte
# Berechnung der Moving Averages
df["MA_Day"] = df["water_level"].rolling(window=minutes_per_day, min_periods=1).mean()
df["MA_Week"] = df["water_level"].rolling(window=minutes_per_week, min_periods=1).mean()
df["MA_Month"] = df["water_level"].rolling(window=minutes_per_month, min_periods=1).mean()
# Spalten neu anordnen (Moving Averages an den Anfang)
df = df[["MA_Month", "MA_Week", "MA_Day"] + df.columns[:-3].tolist()]

array = df.to_numpy()


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


np.save("input", input_stand * 1000) #für app.py
np.save("live_prediction", arr * 1000) #für app.py

#vorhersage für den zweiten plot, wöchentliche vorhersage:

temp_arr = []
historic_prediction = []
historic_prediction_temp = []
historic_prediction_full = []
range_loop = 15 if datetime.now().hour < 12 else 16


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

    historic_prediction_full = np.append(historic_prediction_full, historic_prediction)

historic_prediction_full = historic_prediction_full[:(len(historic_prediction_full)-(720-cut_off_var))]

historic_input_full = array[14 * 60 : 14 * 60 + 12 * 60 * range_loop , 12]

fehler_array = np.abs(historic_prediction_full - historic_input_full)


#nur zum test lokal
import matplotlib.pyplot as plt

# Beispielarray mit 15120 Elementen (hier einfach eine Zahlenreihe von 0 bis 15119)
data = list(range(15120))
plt.figure(figsize=(10, 4))
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

np.save("test_arr", array )
