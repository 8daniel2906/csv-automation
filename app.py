import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

timestamps_array = np.load('timestamps.npy', allow_pickle=True)

# **Konvertiere den ersten Timestamp zu einem datetime-Objekt**
first_timestamp = pd.to_datetime(timestamps_array[1])
second_timestamp = pd.to_datetime(timestamps_array[0])


array_blue = np.load('input.npy')
array_red = np.load('live_prediction.npy')

# **Zeitstempel für die x-Achse**
time_blue = pd.date_range(start=first_timestamp, periods=len(array_blue), freq='T')
time_red = pd.date_range(start=time_blue[-1] + pd.Timedelta(minutes=1), periods=len(array_red), freq='T')
# Erster Plot: Zwei Arrays hintereinander
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=time_blue, y=array_blue, mode='lines', name='Wasserstand der letzten 14 Stunden', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=time_red, y=array_red, mode='lines', name='Vorhersage der nächsten 12 Stunden', line=dict(color='red')))

fig1.update_layout(
    title='Live - Vorhersage',
    xaxis_title='Zeit',
    yaxis_title='Wasserstand in cm',
    xaxis=dict(type='date'),
    dragmode='zoom',
    showlegend=True
)

st.plotly_chart(fig1)

# Eine Trennlinie für bessere Übersicht
st.markdown("---")

# Laden der fünf neuen Arrays für den zweiten Plot
Wasserstand_der_letzten_Woche = np.load('historic_data.npy')
zwölfstündige_Vorhersagen = np.load('historic_predictions.npy')
Fehler_pro_Messung = np.load('error.npy')
array_mean = np.load('mean_error.npy')
array_max = np.load('max_global_error.npy')

# **Erzeuge Zeitstempel für Plot 2**
time_1 = pd.date_range(start=second_timestamp, periods=len(Wasserstand_der_letzten_Woche), freq='T')
# Zweiter Plot: 5 Arrays
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=time_1, y=Wasserstand_der_letzten_Woche, mode='lines', name='Wasserstand der letzten Woche', line=dict(color='blue)))
fig2.add_trace(go.Scatter(x=time_1, y=zwölfstündige_Vorhersagen, mode='lines', name='Wasserstand der letzten Woche', line=dict(color='red')))
fig2.add_trace(go.Scatter(x=time_1, y=Fehler_pro_Messung, mode='lines', name='Wasserstand der letzten Woche', line=dict(color='orange')))

fig2.update_layout(
    title='Wasserstandvorhersage der letzten Woche',
    xaxis_title='Zeit',
    yaxis_title='Wasserstand in cm',
    xaxis=dict(type='date'),
    dragmode='zoom',
    showlegend=True
)

st.plotly_chart(fig2)