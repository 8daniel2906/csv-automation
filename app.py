import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Beispiel: Zwei np.arrays laden
# Ersetze dies durch den Pfad zu deinen eigenen .npy Dateien
array_blue = np.load('input.npy')
array_red = np.load('live_prediction.npy')

# Erstelle eine leere Figure
fig = go.Figure()

# Füge das erste Array (blaue Linie) hinzu
fig.add_trace(go.Scatter(x=np.arange(len(array_blue)), y=array_blue, mode='lines', name='Blau', line=dict(color='blue')))

# Füge das zweite Array (rote Linie) hinzu, wobei der x-Wert für die rote Linie angepasst wird
fig.add_trace(go.Scatter(x=np.arange(len(array_blue), len(array_blue) + len(array_red)),
                         y=array_red, mode='lines', name='Rot', line=dict(color='red')))

# Achsentitel und Layout
fig.update_layout(
    title='Interaktives Plot mit zwei Arrays hintereinander',
    xaxis_title='Index',
    yaxis_title='Wert',
    dragmode='zoom',  # Ermöglicht das Strecken/Verkürzen der Achsen
    showlegend=True
)

# Zeige das Plot in Streamlit an
st.plotly_chart(fig)
