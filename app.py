import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, time

import numpy as np

def convert_json_row_to_arrays(zeile_json: dict):
    """
    Wandelt eine Zeile aus dem JSON (Dict) um, indem Listen in NumPy-Arrays konvertiert werden.

    Args:
        zeile_json (dict): Eine einzelne Zeile aus dem JSON-Response.

    Returns:
        dict: Zeile mit NumPy-Arrays.
    """

    # Kopiere die Zeile, damit das Original nicht verändert wird
    zeile = zeile_json.copy()

    # Konvertiere explizit die Felder
    zeile["historic_prediction"] = np.array(zeile["historic_prediction"])
    zeile["blaue_kurve"] = np.array(zeile["blaue_kurve"])
    zeile["lower_hist"] = np.array(zeile["lower_hist"])
    zeile["upper_hist"] = np.array(zeile["upper_hist"])

    return zeile


def get_live_data():
    response = requests.get("http://127.0.0.1:8000/get-live")
    response_json = response.json()
    converted = convert_json_row_to_arrays(response_json["results"][0])
    return converted

converted = get_live_data()

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

first_timestamp = converted["zeit1"]
second_timestamp = converted["zeit2"]
array_red = converted["historic_prediction"]
array_blue = converted["blaue_kurve"]
upper = converted["upper_hist"]
lower = converted["lower_hist"]

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
    name=' 80% Prädiktionsintervall'
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



# Erzeuge Zeitstempel für Plot 2
time_1 = pd.date_range(start=second_timestamp, periods=len(Wasserstand_der_letzten_Woche), freq='T')

fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=time_1, y=Wasserstand_der_letzten_Woche, mode='lines', name='Wasserstand der letzten Woche', line=dict(color='#66B2FF', width = 0.7)))#farbwahl ist ein helleres blau, da streamlit dunkel ist, wegen kontrast
fig2.add_trace(go.Scatter(x=time_1, y=zwölfstündige_Vorhersagen, mode='lines', name='Vorhersage', line=dict(color='red', width = 0.5)))
fig2.add_trace(go.Scatter(x=time_1, y=Fehler_pro_Messung, mode='lines',
                          name=f'Vorhersagefehler <br>Ø Vorhersagefehler: {array_mean[0]:.2f}cm <br>MAX. Vorhersagefehler: {array_max[0]:.2f}cm',
                          line=dict(color='orange', width = 0.5)))


# Zuerst die untere Linie zeichnen (kein Fill)
fig2.add_trace(go.Scatter(
    x=time_1,
    y=lower_historic,
    mode='lines',
    name=f'Untergrenze 80%-Intervall',
    line=dict(color='rgba(0, 255, 0, 0)'),  # Unsichtbare Linie
    showlegend=False
))

# Dann die obere Linie, die zum unteren Bereich füllt
fig2.add_trace(go.Scatter(
    x=time_1,
    y=upper_historic,
    mode='lines',
    name=f'80%-Prädiktionsintervall <br>Hit-Rate: {np.round(hit_rate*100,2)}% <br>Ø Intervallbreite: {summe}cm',
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
##############################################################################################

st.title("Zeitraum-Auswahl")

# Startdatum & -zeit
start_date = st.date_input("Startdatum", value=datetime.today())
start_hour = st.number_input("Startstunde (0-23)", min_value=0, max_value=23, value=0, step=1)

# Enddatum & -zeit
end_date = st.date_input("Enddatum", value=datetime.today())
end_hour = st.number_input("Endstunde (0-23)", min_value=0, max_value=23, value=23, step=1)

# Kombiniere Datum und Stunde zu datetime-Objekten
start_datetime = datetime.combine(start_date, time(start_hour, 0))
end_datetime = datetime.combine(end_date, time(end_hour, 0))

st.write(f"Startzeit: {start_datetime}")
st.write(f"Endzeit: {end_datetime}")

if st.button('Excel-File downloaden für den Zeitraum'):
    # Zeiten im ISO-Format als String
    start_iso = start_datetime.isoformat()
    end_iso = end_datetime.isoformat()

    json_data = {"start_iso": start_iso, "end_iso": end_iso}

    response = requests.post('https://image-api-latest-1.onrender.com/download-excel', json=json_data)

    if response.status_code == 200:
        st.download_button(
            label="Excel-Datei herunterladen",
            data=response.content,
            file_name="report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error(f"Fehler beim Laden der Datei: {response.status_code}")
        ###########################################################################################

