import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Raw-Link zur CSV-Datei
csv_url = "https://raw.githubusercontent.com/8daniel2906/csv-automation/main/sensor_data.csv"


# Lade die CSV-Datei und überprüfe den Status
response = requests.get(csv_url)

if response.status_code == 200:
    st.text("CSV-Datei erfolgreich geladen!")
    st.code(response.text[:500])  # Zeigt die ersten 500 Zeichen der Datei an
else:
    st.error(f"Fehler beim Laden der Datei: {response.status_code}")

# Streamlit App-Titel
st.title("Live TEST CSV-Plot aus GitHub")

@st.cache_data
def load_data(csv_url):
    # Lade die CSV-Datei von der Raw-URL
    return pd.read_csv(csv_url, delimiter=",", names=["Zeit", "Wert"], skiprows=1)

# Lade die Daten
df = load_data(csv_url)

# Zeitspalte in Datetime umwandeln
df["Zeit"] = pd.to_datetime(df["Zeit"])

# Setze den Zeitstempel als Index
df.set_index("Zeit", inplace=True)

# Generiere eine vollständige Zeitreihe für alle Minuten im gewünschten Bereich
start_time = df.index.min()
end_time = df.index.max()
full_time_index = pd.date_range(start=start_time, end=end_time, freq="T")

# Reindexiere den DataFrame, um alle Minuten abzudecken
df = df.reindex(full_time_index)

# Führe die lineare Interpolation für fehlende Werte durch
df["Wert"] = df["Wert"].interpolate(method="linear")

# Plot erstellen
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["Wert"], marker="o", linestyle="-")
ax.set_xlabel("Zeit")
ax.set_ylabel("Wert")
ax.set_title("Werte über Zeit (mit Interpolation)")
ax.grid(True)
plt.xticks(rotation=45)

# Den Plot in Streamlit anzeigen
st.pyplot(fig)
