import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, time
import psycopg2 as psy
from utils import get_latest_endzeitpunkt_iso, get_earliest_startzeitpunkt_iso

# ------------------ Datenkonvertierung ------------------
def convert_json_row_to_arrays1(zeile_json: dict):
    zeile = zeile_json.copy()
    for key in ["historic_prediction", "blaue_kurve", "lower_hist", "upper_hist"]:
        zeile[key] = np.array(zeile[key])
    return zeile

def convert_json_row_to_arrays2(grouped_json: dict):
    converted = grouped_json.copy()
    for key in ["historic_prediction", "historic_vergleich", "lower_hist", "upper_hist"]:
        converted[key] = [np.array(arr) for arr in grouped_json[key]]
    converted["statistics"] = grouped_json["statistics"]
    return converted

# ------------------ API-Calls ------------------
def get_live_data():
    response = requests.get("https://image-api-latest-3.onrender.com/get-live")
    response_json = response.json()
    return convert_json_row_to_arrays1(response_json["results"][0])

def get_live_data2():
    response = requests.get("https://image-api-latest-3.onrender.com/get-live2")
    response_json = response.json()
    return convert_json_row_to_arrays2(response_json)


# ------------------ Berechnungen ------------------
def calculate_prediction_text(upper, threshold):
    if np.max(upper) > threshold:
        return "<span style='color:red'>Hochwasserwarnung für die nächsten 12 Stunden!</span>"
    else:
        return "<span style='color:green'>Entwarnung für Hochwasser für die nächsten 12 Stunden!</span>"

def calculate_statistics(Wasserstand, Vorhersage, lower_hist, upper_hist):
    error = np.abs(Wasserstand - Vorhersage)
    mean_error = np.mean(error)
    max_error = np.max(error)
    inside = (Wasserstand >= lower_hist) & (Wasserstand <= upper_hist)
    hit_rate = np.sum(inside) / len(Wasserstand)
    avg_interval = np.mean(upper_hist - lower_hist)
    return error, mean_error, max_error, hit_rate, avg_interval

# ------------------ Plots ------------------
def plot_live_prediction(time_blue, array_blue, time_red, array_red, upper, lower, threshold_line):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_blue, y=array_blue, mode='lines', name='Wasserstand letzte 14h', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time_red, y=array_red, mode='lines', name='Vorhersage nächste 12h', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=np.concatenate((time_blue, time_red)), y=threshold_line, mode='lines', name='Hochwasserschwelle', line=dict(color='purple', width=0.7)))

    fig.add_trace(go.Scatter(x=time_red, y=upper, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=time_red, y=lower, mode='lines', fill='tonexty', fillcolor='rgba(255,165,0,0.3)', line=dict(width=0), name='80% Prädiktionsintervall'))

    fig.update_layout(title='Live - Vorhersage', xaxis_title='Zeit', yaxis_title='Wasserstand in cm', xaxis=dict(type='date'), dragmode='zoom', showlegend=True)
    return fig

def plot_week_prediction(time, actual, forecast, error, lower, upper, threshold_line, mean_error, max_error, hit_rate, avg_interval):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=actual, mode='lines', name='Wasserstand Woche', line=dict(color='#66B2FF', width=0.7)))
    fig.add_trace(go.Scatter(x=time, y=forecast, mode='lines', name='Vorhersage', line=dict(color='red', width=0.5)))
    fig.add_trace(go.Scatter(x=time, y=error, mode='lines', name=f'Fehler Ø: {mean_error:.2f}cm, MAX: {max_error:.2f}cm', line=dict(color='orange', width=0.5)))
    fig.add_trace(go.Scatter(x=time, y=threshold_line, mode='lines', name='Hochwasserschwelle', line=dict(color='purple', width=0.5)))

    fig.add_trace(go.Scatter(x=time, y=lower, mode='lines', line=dict(color='rgba(0,255,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=upper, mode='lines', name=f'80%-Intervall Hit-Rate: {hit_rate*100:.2f}%, Ø Breite: {avg_interval:.1f}cm', line=dict(color='green', width=0.1), fill='tonexty', fillcolor='rgba(0,255,0,0.2)'))

    fig.update_layout(title='Wasserstandvorhersage der letzten Woche',
                      xaxis_title='Zeit',
                      yaxis_title='Wasserstand in cm',
                      xaxis=dict(type='date', rangeslider=dict(visible=True),
                                 rangeselector=dict(buttons=[
                                     dict(count=1, label="1T", step="day", stepmode="backward"),
                                     dict(count=3, label="3T", step="day", stepmode="backward"),
                                     dict(step="all")
                                 ])),
                      dragmode='zoom', showlegend=True)
    return fig

# ------------------ Main UI ------------------
def main():
    hochwasser_schwelle = 750

    # Daten laden
    converted = get_live_data()
    converted2 = get_live_data2()

    # Live-Daten
    time_blue = pd.date_range(start=converted["zeit1"], periods=len(converted["blaue_kurve"]), freq='T')
    time_red = pd.date_range(start=time_blue[-1] + pd.Timedelta(minutes=1), periods=len(converted["historic_prediction"]), freq='T')
    threshold_line_live = np.zeros(len(time_blue) + len(time_red)) + hochwasser_schwelle

    st.markdown(calculate_prediction_text(converted["upper_hist"], hochwasser_schwelle), unsafe_allow_html=True)
    st.plotly_chart(plot_live_prediction(time_blue, converted["blaue_kurve"], time_red, converted["historic_prediction"], converted["upper_hist"], converted["lower_hist"], threshold_line_live))

    st.markdown("---")

    # Vergangenheitsdaten
    Wasserstand = np.array(converted2["historic_vergleich"]).flatten()
    Vorhersage = np.array(converted2["historic_prediction"]).flatten()
    lower_hist = np.array(converted2["lower_hist"]).flatten()
    upper_hist = np.array(converted2["upper_hist"]).flatten()

    time_week = pd.date_range(start=converted2["zeit1"][0], periods=len(Wasserstand), freq='T')
    threshold_line_week = np.zeros(len(Wasserstand)) + hochwasser_schwelle

    error, mean_error, max_error, hit_rate, avg_interval = calculate_statistics(Wasserstand, Vorhersage, lower_hist, upper_hist)

    stats = converted2["statistics"]

    st.markdown(f"""
    **Ergebnisse der letzten Woche:**

    - ✅ Warnung und Hochwasser: **{stats[0]}**
    - ❌ Entwarnung, aber Hochwasser: **{stats[1]}**
    - ❌ Warnung, aber kein Hochwasser: **{stats[2]}**
    - ✅ Entwarnung und kein Hochwasser: **{stats[3]}**
    """)

    st.plotly_chart(plot_week_prediction(time_week, Wasserstand, Vorhersage, error, lower_hist, upper_hist, threshold_line_week, mean_error, max_error, hit_rate, avg_interval))

    st.markdown("---")

    # Zeitraum-Auswahl & Download
    st.title("Zeitraum-Auswahl")
    col1, col2 = st.columns([1, 5])  # Erste Spalte schmaler (1), zweite breiter (5)

    with col1:

        start_date = st.date_input("Startdatum", value=datetime.today())
        start_hour = st.number_input("Startstunde (0-23)", min_value=0, max_value=23, value=0)
        end_date = st.date_input("Enddatum", value=datetime.today())
        end_hour = start_hour

    start_datetime = datetime.combine(start_date, time(start_hour, 0))
    end_datetime = datetime.combine(end_date, time(end_hour, 0))

    st.write(f"Startzeit: {start_datetime}")
    st.write(f"Endzeit: {end_datetime}")

    if st.button("Excel-File downloaden für den Zeitraum"):
        json_data = {"start_iso": start_datetime.isoformat(), "end_iso": end_datetime.isoformat()}
        #response = requests.post("http://127.0.0.1:8000/download-excel", json=json_data)
        response = requests.post("https://image-api-latest-3.onrender.com/download-excel", json=json_data)
        if response.status_code == 200:
            st.download_button(
                label="Excel-Datei herunterladen",
                data=response.content,
                file_name="report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error(f"Fehler beim Laden der Datei: {response.status_code}")

if __name__ == "__main__":
    main()
