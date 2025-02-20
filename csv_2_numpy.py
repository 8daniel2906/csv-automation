import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError





# CSV einlesen
df = pd.read_csv("sensor_data.csv", delimiter=",", names=["Zeit", "Wert"], skiprows=1)

# Zeitspalte in Datetime umwandeln
df["Zeit"] = pd.to_datetime(df["Zeit"])

# Neue Spalten für Jahr, Monat, Tag, Stunde, Minute hinzufügen
df["Jahr"] = df["Zeit"].dt.year
df["Monat"] = df["Zeit"].dt.month
df["Tag"] = df["Zeit"].dt.day
df["Stunde"] = df["Zeit"].dt.hour
df["Minute"] = df["Zeit"].dt.minute

# Unnötige Zeitspalte entfernen
df.drop(columns=["Zeit"], inplace=True)

# Spalten neu anordnen, sodass "Wert" die letzte Spalte ist
df = df[["Jahr", "Monat", "Tag", "Stunde", "Minute", "Wert"]]

# Fehlende Zeitwerte ergänzen und interpolieren
start_time = df.iloc[0][["Jahr", "Monat", "Tag", "Stunde", "Minute"]].tolist()
end_time = df.iloc[-1][["Jahr", "Monat", "Tag", "Stunde", "Minute"]].tolist()
full_time_index = pd.date_range(
    start=f"{start_time[0]}-{start_time[1]:02d}-{start_time[2]:02d} {start_time[3]:02d}:{start_time[4]:02d}",
    end=f"{end_time[0]}-{end_time[1]:02d}-{end_time[2]:02d} {end_time[3]:02d}:{end_time[4]:02d}",
    freq="T"
)

# Erstelle einen neuen DataFrame mit der vollständigen Zeitreihe
df_full = pd.DataFrame(
    [(t.year, t.month, t.day, t.hour, t.minute) for t in full_time_index],
    columns=["Jahr", "Monat", "Tag", "Stunde", "Minute"]
)

# Werte mit ursprünglichem DataFrame zusammenführen und fehlende Werte interpolieren
df = df_full.merge(df, on=["Jahr", "Monat", "Tag", "Stunde", "Minute"], how="left")
df["Wert"] = df["Wert"].interpolate(method="linear")

# Letzte 900 Zeilen auswählen
df_last_900 = df.tail(900)

import numpy as np

#numpy_array = df_last_900.to_numpy()


def transform_and_append(df):
    # Nehme die erste Zeile aus den 900
    first_row = df.iloc[0]

    # Extrahiere Jahr, Monat, Tag, Stunde, Minute
    year = first_row["Jahr"] - 2000
    year = year / 1000
    month = first_row["Monat"]
    day = first_row["Tag"]
    hour = first_row["Stunde"]
    minute = first_row["Minute"]

    # Zyklische Sin/Cos-Transformation
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    # Erstelle ein Array mit den 9 Datenpunkten
    transformed_data = np.array([
        year, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, minute_sin, minute_cos
    ])

    # Werte der 900 Zeilen als Numpy-Array
    values = df["Wert"].to_numpy() / 1000

    # Kombiniere die 9 Features mit den 900 Werten zu einem 909-Array
    result_array = np.concatenate([transformed_data, values])

    return result_array


# Die letzten 900 Zeilen nehmen und die Funktion anwenden
final_array_for_model_input = transform_and_append(df_last_900)
final_array_for_input_plot = final_array_for_model_input[9:]

model = tf.keras.models.load_model(
    "fnn_v2.h5",
    custom_objects={"mse": MeanSquaredError()}
)
next_prediction = model.predict(final_array_for_model_input.reshape(1, -1), verbose=0).reshape(180,)

print(final_array_for_input_plot.reshape(900,).shape)
print(next_prediction.shape)
arr = []
arr = np.concatenate([final_array_for_input_plot, next_prediction ])
#np.save("test", final_array_for_model_input)
import matplotlib.pyplot as plt

# Plotten des Arrays
plt.figure(figsize=(10, 6))  # Optional: Größe des Plots anpassen
plt.plot(arr * 1000)
plt.title('Plot der 1080 Werte')
plt.xlabel('Index')
plt.ylabel('Wert')
plt.grid(True)  # Optional: Gitterlinien hinzufügen
plt.show()
