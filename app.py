import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import requests

csv_url = "https://github.com/8daniel2906/csv-automation/blob/9fb2c6fe5f2ee5c1e10ae5344d48d88b9c3dbafc/sensor_data.csv"
response = requests.get(csv_url)

if response.status_code == 200:
    st.text("CSV-Datei erfolgreich geladen!")
    st.code(response.text[:500])  # Zeigt die ersten 500 Zeichen
else:
    st.error(f"Fehler beim Laden der Datei: {response.status_code}")

# Streamlit App-Titel
st.title("Live CSV-Plot aus GitHub")

# CSV-Datei aus GitHub laden
csv_url = "https://github.com/8daniel2906/csv-automation/blob/9fb2c6fe5f2ee5c1e10ae5344d48d88b9c3dbafc/sensor_data.csv"  # Ersetze mit deinem Dateinamen

@st.cache_data
def load_data(url):
    return pd.read_csv(url, delimiter=r"\s+", names=["Zeit", "Wert"], skiprows=1)

# Lade die Daten
df = load_data(csv_url)

# Zeitspalte in Datetime umwandeln
df["Zeit"] = pd.to_datetime(df["Zeit"])

# Zeige die Daten in der App
st.write("### Vorschau der Daten")
st.dataframe(df.head())

# Plot erstellen
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Zeit"], df["Wert"], marker="o", linestyle="-")
ax.set_xlabel("Zeit")
ax.set_ylabel("Wert")
ax.set_title("Werte über Zeit")
ax.grid(True)
plt.xticks(rotation=45)

# Den Plot in Streamlit anzeigen
st.pyplot(fig)
