import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

timestamps_array = np.load('timestamps.npy', allow_pickle=True)

# **Konvertiere den ersten Timestamp zu einem datetime-Objekt**
first_timestamp = pd.to_datetime(timestamps_array[1])
second_timestamp = pd.to_datetime(timestamps_array[0])

# **Manuelle Eingabe des Start-Zeitpunkts als Text**
datetime_str = st.text_input("Gib das Startdatum & Uhrzeit ein (Format: YYYY-MM-DD HH:MM)", value=first_timestamp.strftime("%Y-%m-%d %H:%M"))

# **Konvertiere den Text in einen Timestamp**
try:
    start_datetime = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M")
except ValueError:
    st.error(" Falsches Format! Bitte nutze YYYY-MM-DD HH:MM (z. B. 2024-02-21 14:30).")
    st.stop()

# Beispiel: Zwei np.arrays laden
# Ersetze dies durch den Pfad zu deinen eigenen .npy Dateien

array_blue = np.load('input.npy')
array_red = np.load('live_prediction.npy')

# **Zeitstempel für die x-Achse**
time_blue = pd.date_range(start=start_datetime, periods=len(array_blue), freq='T')
time_red = pd.date_range(start=time_blue[-1] + pd.Timedelta(minutes=1), periods=len(array_red), freq='T')
# Erster Plot: Zwei Arrays hintereinander
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=time_blue, y=array_blue, mode='lines', name='Wasserstand_der_letzten_Stunden', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=time_red, y=array_red, mode='lines', name='Vorhersage', line=dict(color='red')))

fig1.update_layout(
    title='Interaktives Zeit-Plot mit zwei Arrays',
    xaxis_title='Zeit',
    yaxis_title='Wert',
    xaxis=dict(type='date'),
    dragmode='zoom',
    showlegend=True
)

st.plotly_chart(fig1)

# Eine Trennlinie für bessere Übersicht
st.markdown("---")

# **Manuelle Eingabe für Plot 2**
datetime_str2 = st.text_input("Gib das Startdatum & Uhrzeit für Plot 2 ein (Format: YYYY-MM-DD HH:MM)", value=second_timestamp.strftime("%Y-%m-%d %H:%M"))
# **Konvertiere Eingabe in Timestamp für Plot 2**
try:
    start_datetime2 = pd.to_datetime(datetime_str2, format="%Y-%m-%d %H:%M")
except ValueError:
    st.error("❌ Falsches Format für Plot 2! Bitte nutze YYYY-MM-DD HH:MM.")
    st.stop()

# Laden der fünf neuen Arrays für den zweiten Plot
Wasserstand_der_letzten_Woche = np.load('historic_data.npy')
zwölfstündige_Vorhersagen = np.load('historic_predictions.npy')
Fehler_pro_Messung = np.load('error.npy')
array_4 = np.load('mean_error.npy')
array_5 = np.load('max_global_error.npy')

# Farben für die Linien
colors = ['purple', 'green', 'orange', 'cyan', 'magenta']
arrays = [Wasserstand_der_letzten_Woche, zwölfstündige_Vorhersagen, Fehler_pro_Messung, array_4, array_5]

# **Erzeuge Zeitstempel für Plot 2**
time_steps = [pd.date_range(start=start_datetime2, periods=len(arr), freq='T') for arr in arrays]

# Zweiter Plot: 5 Arrays
fig2 = go.Figure()



for i, (array, time, color) in enumerate(zip(arrays, time_steps, colors)):
    fig2.add_trace(go.Scatter(x=time, y=array, mode='lines', name=f'Array {i+1}', line=dict(color=color)))

fig2.update_layout(
    title='Interaktives Zeit-Plot 2 mit fünf Arrays',
    xaxis_title='Zeit',
    yaxis_title='Wert',
    xaxis=dict(type='date'),
    dragmode='zoom',
    showlegend=True
)

st.plotly_chart(fig2)