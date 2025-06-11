import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

timestamps_array = np.load('timestamps.npy', allow_pickle=True)

# Konvertiere den ersten Timestamp zu einem datetime-Objekt
first_timestamp = pd.to_datetime(timestamps_array[1])
second_timestamp = pd.to_datetime(timestamps_array[0])


array_blue = np.load('input.npy')
array_red = np.load('live_prediction.npy')
lower = np.load('lower.npy')
upper = np.load('upper.npy')
lower_historic = np.load('lower_historic.npy')
upper_historic = np.load('upper_historic.npy')

# Zeitstempel für die x-Achse
time_blue = pd.date_range(start=first_timestamp, periods=len(array_blue), freq='T')
time_red = pd.date_range(start=time_blue[-1] + pd.Timedelta(minutes=1), periods=len(array_red), freq='T')
# Erster Plot: Zwei Arrays hintereinander
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=time_blue, y=array_blue, mode='lines', name='Wasserstand der letzten 14 Stunden', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=time_red, y=array_red, mode='lines', name='Vorhersage der nächsten 12 Stunden', line=dict(color='red')))
# Obere Grenze
fig1.add_trace(go.Scatter(
    x=time_red,
    y=upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False
))

# Untere Grenze mit Füllung bis zur oberen
fig1.add_trace(go.Scatter(
    x=time_red,
    y=lower,
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255,165,0,0.3)',  # Orange mit alpha 0.3
    line=dict(width=0),
    name='± 90% Quantil'
))

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
array_mean = np.load('mean_error.npy') #alle werte hier sind identisch
array_max = np.load('max_global_error.npy')#alle werte hier sind identisch
print(len(Wasserstand_der_letzten_Woche))
print(len(lower_historic))
# 🔸 Trefferquote berechnen
inside = (Wasserstand_der_letzten_Woche >= lower_historic) & (Wasserstand_der_letzten_Woche <= upper_historic)
count_inside = np.sum(inside)
total = len(Wasserstand_der_letzten_Woche)
hit_rate = count_inside / total
summe = np.sum(upper_historic - lower_historic)/total
print('hitrate', hit_rate)
print("summe", summe)


# Erzeuge Zeitstempel für Plot 2
time_1 = pd.date_range(start=second_timestamp, periods=len(Wasserstand_der_letzten_Woche), freq='T')

fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=time_1, y=Wasserstand_der_letzten_Woche, mode='lines', name='Wasserstand der letzten Woche', line=dict(color='#66B2FF', width = 0.7)))#farbwahl ist ein helleres blau, da streamlit dunkel ist, wegen kontrast
fig2.add_trace(go.Scatter(x=time_1, y=zwölfstündige_Vorhersagen, mode='lines', name='Vorhersage', line=dict(color='red', width = 0.5)))
fig2.add_trace(go.Scatter(x=time_1, y=Fehler_pro_Messung, mode='lines',
                          name=f'Vorhersagefehler <br>Ø Vorhersagefehler: {array_mean[0]:.2f}cm <br>MAX. Vorhersagefehler: {array_max[0]:.2f}cm',
                          line=dict(color='orange', width = 0.5)))

#fig2.add_trace(go.Scatter(
  #  x=time_1,
  #  y=lower_historic,
  #  mode='lines',
  #  name=f'Obergrenze 90%-Intervall <br> Hit-Rate: {hit_rate}%',  # Optional: beschreibe die Linie sinnvoll
  #  line=dict(color='green', width=0.1)
#))

#fig2.add_trace(go.Scatter(
 #   x=time_1,
 #   y=upper_historic,
 #   mode='lines',
 #   name=f'Untergrenze 90%-Intervall <br> Ø Intervallbreite: {Summe}cm',  # Optional: beschreibe die Linie sinnvoll
 #   line=dict(color='green', width=0.1)
#))
# Zuerst die untere Linie zeichnen (kein Fill)
fig2.add_trace(go.Scatter(
    x=time_1,
    y=lower_historic,
    mode='lines',
    name=f'Untergrenze 90%-Intervall',
    line=dict(color='rgba(0, 255, 0, 0)'),  # Unsichtbare Linie
    showlegend=False
))

# Dann die obere Linie, die zum unteren Bereich füllt
fig2.add_trace(go.Scatter(
    x=time_1,
    y=upper_historic,
    mode='lines',
    name=f'90%-Konfidenzintervall <br>Hit-Rate: {hit_rate}% <br>Ø Intervallbreite: {summe}cm',
    line=dict(color='green', width=0.1),
    fill='tonexty',
    fillcolor='rgba(0, 255, 0, 0.2)'  # Transparente grüne Füllung
))

# hinzufügen eines range sliders
fig2.update_layout(
    title='Wasserstandvorhersage der letzten Woche',
    xaxis_title='Zeit',
    yaxis_title='Wasserstand in cm',
    xaxis=dict(
        type='date',
        rangeslider=dict(visible=True),  # Range Slider aktivieren
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1T", step="day", stepmode="backward"),
                dict(count=3, label="3T", step="day", stepmode="backward"),
                dict(step="all")
            ]
        )
    ),
    dragmode='zoom',
    showlegend=True
)

st.plotly_chart(fig2)