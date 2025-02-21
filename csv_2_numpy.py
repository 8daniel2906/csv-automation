import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
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

print(arr.shape)


#arr = np.concatenate([input_stand, arr ])
np.save("input", input_stand * 1000)
np.save("live_prediction", arr * 1000)
