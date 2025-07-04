import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, time


def convert_json_row_to_arrays1(zeile_json: dict):
    zeile = zeile_json.copy()
    zeile["historic_prediction"] = np.array(zeile["historic_prediction"])
    zeile["blaue_kurve"] = np.array(zeile["blaue_kurve"])
    zeile["lower_hist"] = np.array(zeile["lower_hist"])
    zeile["upper_hist"] = np.array(zeile["upper_hist"])
    return zeile

def convert_json_row_to_arrays2(grouped_json: dict):
    converted = grouped_json.copy()

    converted["historic_prediction"] = [np.array(arr) for arr in grouped_json["historic_prediction"]]
    converted["historic_vergleich"] = [np.array(arr) for arr in grouped_json["historic_vergleich"]]
    converted["lower_hist"] = [np.array(arr) for arr in grouped_json["lower_hist"]]
    converted["upper_hist"] = [np.array(arr) for arr in grouped_json["upper_hist"]]
    converted["statistics"] = grouped_json["statistics"]

    return converted

def get_live_data():
    response = requests.get("https://image-api-latest-3.onrender.com/get-live")
    #response = requests.get("http://127.0.0.1:8000/get-live")
    response_json = response.json()
    converted = convert_json_row_to_arrays1(response_json["results"][0])
    return converted

def get_live_data2():
    response = requests.get("https://image-api-latest-3.onrender.com/get-live2")
    #response = requests.get("http://127.0.0.1:8000/get-live2")
    response_json = response.json()
    converted = convert_json_row_to_arrays2(response_json)
    return converted

def vorhersage_live_text(upper):
    vorhersage_text = ["<span style='color:red'>Hochwasserwarnung für die nächsten 12 Stunden!</span>",
                       "<span style='color:green'>Entwarnung für Hochwasser für die nächsten 12 Stunden!</span>"]
    return vorhersage_text[0] if np.max(upper) > hochwasser_schwelle else vorhersage_text[1]



converted = get_live_data()
converted2 = get_live_data2()


lower_historic = np.load('lower_historic.npy')
upper_historic = np.load('upper_historic.npy')

first_timestamp = converted["zeit1"]
second_timestamp = converted["zeit2"]
array_red = converted["historic_prediction"]
array_blue = converted["blaue_kurve"]
upper = converted["upper_hist"]
lower = converted["lower_hist"]
statistics = np.array(converted2["statistics"])



first_timestamp2 = converted2["zeit1"][0]
second_timestamp2 = converted2["zeit2"][-2]
zwölfstündige_Vorhersagen = np.array(converted2["historic_prediction"]).flatten()
Wasserstand_der_letzten_Woche = np.array(converted2["historic_vergleich"]).flatten()
upper_historic = np.array(converted2["upper_hist"]).flatten()
lower_historic = np.array(converted2["lower_hist"]).flatten()
hochwasser_schwelle = 750
schwelle_plot = np.zeros(len(Wasserstand_der_letzten_Woche)) + hochwasser_schwelle
# Zeitstempel für die x-Achse
time_blue = pd.date_range(start=first_timestamp, periods=len(array_blue), freq='T')
time_red = pd.date_range(start=time_blue[-1] + pd.Timedelta(minutes=1), periods=len(array_red), freq='T')

st.markdown(vorhersage_live_text(upper), unsafe_allow_html=True)
# Erster Plot: Zwei Arrays hintereinander
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=time_blue, y=array_blue, mode='lines', name='Wasserstand der letzten 14 Stunden', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=time_red, y=array_red, mode='lines', name='Vorhersage der nächsten 12 Stunden', line=dict(color='red')))
fig1.add_trace(go.Scatter(x=np.concatenate((time_blue, time_red)), y=schwelle_plot, mode='lines', name="Hochwasserschwelle", line=dict(color='purple', width = 0.7)))
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


Fehler_pro_Messung = np.abs(Wasserstand_der_letzten_Woche - zwölfstündige_Vorhersagen)
array_mean = np.mean(Fehler_pro_Messung) #alle werte hier sind identisch
array_max = np.max(Fehler_pro_Messung)#alle werte hier sind identisch

# 🔸 Trefferquote berechnen
inside = (Wasserstand_der_letzten_Woche >= lower_historic) & (Wasserstand_der_letzten_Woche <= upper_historic)
count_inside = np.sum(inside)
total = len(Wasserstand_der_letzten_Woche)
hit_rate = count_inside / total
summe = np.sum(upper_historic - lower_historic)/total




st.markdown(f"""
**Ergebnisse der letzen Woche:**

- ✅ Warnung und Hochwasser  : **{statistics[0]}**
- ❌ Entwarnung, aber Hochwasser : **{statistics[1]}**
- ❌ Warnung, aber kein Hochwasser : **{statistics[2]}**
- ✅ Entwarnung und kein Hochwasser : **{statistics[3]}**
""")

# Erzeuge Zeitstempel für Plot 2
time_1 = pd.date_range(start=first_timestamp2, periods=len(Wasserstand_der_letzten_Woche), freq='T')

fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=time_1, y=Wasserstand_der_letzten_Woche, mode='lines', name='Wasserstand der letzten Woche', line=dict(color='#66B2FF', width = 0.7)))#farbwahl ist ein helleres blau, da streamlit dunkel ist, wegen kontrast
fig2.add_trace(go.Scatter(x=time_1, y=zwölfstündige_Vorhersagen, mode='lines', name='Vorhersage', line=dict(color='red', width = 0.5)))
fig2.add_trace(go.Scatter(x=time_1, y=Fehler_pro_Messung, mode='lines',
                          name=f'Vorhersagefehler <br>Ø Vorhersagefehler: {array_mean:.2f}cm <br>MAX. Vorhersagefehler: {array_max:.2f}cm',
                          line=dict(color='orange', width = 0.5)))
fig2.add_trace(go.Scatter(x=time_1, y=schwelle_plot, mode='lines', name='Hochwasser_schwelle', line=dict(color='purple', width = 0.5)))


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

    response = requests.post('https://image-api-latest-3.onrender.com/download-excel', json=json_data)

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

