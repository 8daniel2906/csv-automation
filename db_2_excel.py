import psycopg2 as psy
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
import pandas as pd
import io

app = FastAPI()

def load_time_series(conn_str, startzeit_iso, endzeit_iso):
    with psy.connect(conn_str) as conn:
        with conn.cursor() as cur:
            current_start = startzeit_iso
            results = []

            while True:
                # Hole den ersten Datensatz, dessen startzeit = current_start ist
                cur.execute("""
                    SELECT id, startzeit, endzeit
                    FROM zeitreihe_metadata
                    WHERE startzeit = %s
                      AND endzeit <= %s
                    ORDER BY endzeit ASC
                    LIMIT 1
                """, (current_start, endzeit_iso))
                row = cur.fetchone()

                if not row:
                    print(f"⚠️ Keine weiteren Zeiträume gefunden ab {current_start}. Abbruch.")
                    break

                metadata_id, startzeit, endzeit = row

                # Hole Zeitreihendaten für dieses metadata_id
                cur.execute("""
                    SELECT minute_value, prediction_value, historic_value, lowerpi_value, upperpi_value
                    FROM zeitreihe_daten
                    WHERE metadata_id = %s
                    ORDER BY minute_value ASC
                """, (metadata_id,))
                data_rows = cur.fetchall()

                # Check, ob wir weniger als 720 Minuten haben
                num_points = len(data_rows)
                if num_points < 144:
                    print(f"⚠️ Zeitreihe bei {startzeit} - {endzeit} hat nur {num_points} Punkte (truncated).")
                elif num_points > 144:
                    print(f"⚠️ Zeitreihe länger als 720 Punkte, wird abgeschnitten.")
                    data_rows = data_rows[:144]

                # Speichere Ergebnis
                results.append({
                    "metadata_id": metadata_id,
                    "startzeit": startzeit,
                    "endzeit": endzeit,
                    "data": data_rows
                })

                # Nächster Startzeitpunkt = aktueller Endzeitpunkt
                current_start = endzeit.strftime("%Y-%m-%dT%H:%M:%S")

                # Prüfen, ob wir über Endzeit hinaus sind
                if current_start >= endzeit_iso:
                    print("✅ Endzeit erreicht oder überschritten. Fertig.")
                    break

            return results



def stretch_array(arr):
    arr = np.array(arr)
    n = len(arr)

    if n < 2:
        raise ValueError("Array muss mindestens 2 Elemente haben, um zu interpolieren.")

    # Für jedes Paar von Punkten wollen wir 5 Punkte insgesamt (Originalpunkte + 4 dazwischen)
    # Z.B. [a, b] -> [a, ..., b] mit 5 Punkten zwischen a und b
    # Wir bauen einen Index-Vektor von 0 bis n-1, jeweils in 1er-Schritten
    old_indices = np.arange(n)

    # Neuer Index-Vektor mit feinerer Auflösung
    new_indices = np.linspace(0, n - 1, (n - 1) * 5 + 1)

    # Interpolation
    new_arr = np.interp(new_indices, old_indices, arr)

    return new_arr.tolist()

def extract_and_stretch(results):
    """Extrahiert Spalten aus einem Block und interpoliert sie."""
    pred_ = []
    hist_ = []
    lower_ = []
    upper_ = []
    for i in range(len(results)):
        pred  = stretch_array([row[1] for row in results[i]['data']])
        hist  = stretch_array([row[2] for row in results[i]['data']])
        lower = stretch_array([row[3] for row in results[i]['data']])
        upper = stretch_array([row[4] for row in results[i]['data']])
        pred_.append(pred)
        hist_.append(hist)
        lower_.append(lower)
        upper_.append(upper)
    return np.array(pred_).flatten(), np.array(hist_).flatten(), np.array(lower_).flatten(), np.array(upper_).flatten()

def plot_time_series(pred, hist, lower, upper):
    """Plottet die Zeitreihen."""

    plt.figure(figsize=(14, 6))
    plt.plot(pred, label="Prediction")
    plt.plot(hist, label="Historic")
    plt.plot(lower, label="Lower PI", linestyle="--")
    plt.plot(upper, label="Upper PI", linestyle="--")
    plt.fill_between(range(len(pred)), lower, upper, alpha=0.2, color='gray')
    plt.legend()
    plt.title("Interpolierte Zeitreihen")
    plt.grid(True)
    plt.show()


def plot_time_series(pred, hist, lower, upper, save_path=None):
    plt.figure(figsize=(14, 6))
    plt.plot(pred, label="Prediction")
    plt.plot(hist, label="Historic")
    plt.plot(lower, label="Lower PI", linestyle="--")
    plt.plot(upper, label="Upper PI", linestyle="--")
    plt.fill_between(range(len(pred)), lower, upper, alpha=0.2, color='gray')
    plt.legend()
    plt.title("Interpolierte Zeitreihen")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"📈 Plot gespeichert unter {save_path}")
    plt.show()


def save_to_excel(pred, hist, lower, upper, filename):
    df = pd.DataFrame({
        "Prediction": pred,
        "Historic": hist,
        "Lower_PI": lower,
        "Upper_PI": upper
    })
    df.index.name = "Index"
    df.to_excel(filename)
    print(f"📄 Daten gespeichert in {filename}")

import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

def export_to_excel_with_chart(pred, hist, lower, upper, filename="zeitreihe.xlsx"):
    # Baue DataFrame
    df = pd.DataFrame({
        "Prediction": pred,
        "Historic": hist,
        "Lower_PI": lower,
        "Upper_PI": upper
    })

    # Speichere erstmal ohne Chart
    df.to_excel(filename, index=False)

    # Lade Workbook mit openpyxl
    wb = load_workbook(filename)
    ws = wb.active

    # Erstelle LineChart
    chart = LineChart()
    chart.title = "Zeitreihe mit Prediction, Historic & PI"
    chart.y_axis.title = "Wert"
    chart.x_axis.title = "Zeitpunkte"

    # Referenz: alle Spalten (A bis D), Zeilen 2 bis letzte
    data = Reference(ws, min_col=1, max_col=4, min_row=1, max_row=ws.max_row)
    chart.add_data(data, titles_from_data=True)

    # Positioniere Chart (z. B. ab Zelle F2)
    ws.add_chart(chart, "F2")

    # Speichern
    wb.save(filename)
    print(f"✅ Excel-Datei mit Diagramm gespeichert: {filename}")
    return filename


conn_str = "postgresql://neondb_owner:npg_mPqZi9CG2txF@ep-divine-mud-a90zxdvg-pooler.gwc.azure.neon.tech/neondb?sslmode=require&channel_binding=require"

conn_str = "postgresql://neondb_owner:npg_mPqZi9CG2txF@ep-divine-mud-a90zxdvg-pooler.gwc.azure.neon.tech/neondb?sslmode=require&channel_binding=require"
start_iso = "2025-06-01T00:00:00"
end_iso = "2025-06-2T00:00:00"

results = np.array(load_time_series(conn_str, start_iso, end_iso))
pred, hist, lower, upper  = extract_and_stretch(results)
plot_time_series(pred, hist, lower, upper, save_path="time_series_plot.png")
excel_path = export_to_excel_with_chart(pred, hist, lower, upper, filename="time_series_data2.xlsx")
print(excel_path)
app = FastAPI()
@app.get("/download-excel")
def download_excel():
    conn_str = "postgresql://neondb_owner:npg_mPqZi9CG2txF@ep-divine-mud-a90zxdvg-pooler.gwc.azure.neon.tech/neondb?sslmode=require&channel_binding=require"
+

    results = np.array(load_time_series(conn_str, start_iso, end_iso))
    pred, hist, lower, upper = extract_and_stretch(results)

    # Statt Dateipfad -> BytesIO
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Beispiel: deine Daten in Excel schreiben
        pd.DataFrame(pred).to_excel(writer, sheet_name='Predictions')
        pd.DataFrame(hist).to_excel(writer, sheet_name='History')
        # du kannst hier beliebig Tabellen hinzufügen

    output.seek(0)  # ganz wichtig!

    return StreamingResponse(
        output,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": "attachment; filename=report.xlsx"}
    )
