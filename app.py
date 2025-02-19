import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Raw-Link zur CSV-Datei
csv_url = "https://raw.githubusercontent.com/8daniel2906/csv-automation/9fb2c6fe5f2ee5c1e10ae5344d48d88b9c3dbafc/sensor_data.csv"

# Lade die CSV-Datei und überprüfe den Status
response = requests.get(csv_url)

if response.status_code == 200:
    st.text("CSV-Datei erfolgreich geladen!")
    st.code(response.text[:500])  # Zeigt die ersten 500 Zeichen der Datei an
else:
    st.error(f"Fehler beim Laden der Datei: {response.status_code}")

# Streamlit App-Titel
st.title("Live CSV-Plot aus GitHub")

@st.cache_data
def load_data(url):
    # Lade die CSV-Datei von der Raw-URL
    return pd.read_csv(url, delimiter=",", names=["Zeit", "Wert"], skiprows=1)

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
