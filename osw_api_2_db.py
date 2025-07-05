import io
from config import api_url_template, conn_str
from utils import *

def inference( array, start):
    #temp_arr = []
    #historic_prediction = []
    #historic_prediction_temp = []
    results = []
    model, kmeans, interval_matrix, chunk, cluster, time_teile, time_labels_klein = get_models()

    steps = len(array)/60 -25

    for i in range(int(steps)):
        historic_datum = array[ i * 60, :12].reshape(12, )
        historic_stand = array[ i * 60 : ( i * 60) + 840, 12].reshape(840, )
        historic_vergleich = array[( i * 60) + 840 : ( i * 60) + 840 + 720, 12]*1000
        temp_arr = np.concatenate((historic_datum, historic_stand))
        historic_prediction = []
        for j in range(2):
            historic_prediction_temp = model.predict(temp_arr.reshape(1, -1), verbose=0).reshape(360, )
            historic_prediction = np.append(historic_prediction, historic_prediction_temp)
            temp_arr = temp_arr[12:]
            temp_arr = np.concatenate((temp_arr, historic_prediction_temp))
            temp_arr = temp_arr[360:]
            temp_arr = np.concatenate((array[ i * 60 + (j + 1) * 360, :12].reshape(12, ), temp_arr))

        historic_prediction = historic_prediction * 1000
        cluster_labels2 = assign_clusters_inference( historic_prediction, chunk, kmeans)  # chunk, kmeans global
        q_low_hist, q_up_hist = get_quantile_bounds_from_labels(cluster_labels2, time_labels_klein, interval_matrix)
        lower_hist = historic_prediction - np.abs(q_low_hist)
        upper_hist = historic_prediction + np.abs(q_up_hist)
        zeit1 = start
        zeit2 = stunden_danach(start, 12)
        start = stunden_danach(start, 1)
        zeile = [zeit1, zeit2, np.max(np.array(historic_vergleich)), np.max(np.array(upper_hist)), np.array(historic_prediction), np.array(historic_vergleich), np.array(lower_hist), np.array(upper_hist)]
        results.append(zeile)

    return results

    return results
def load_in_db2(conn_str, results):
    with psy.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # Vorhandene Kombinationen laden
            cur.execute("SELECT startzeit, endzeit FROM zeitreihe_metadata")
            vorhandene = set(cur.fetchall())

            buffer = io.StringIO()

            for zeile in results:
                zeit1, zeit2, max_vergleich, max_upper, prediction, vergleich, lower, upper = zeile

                if (zeit1, zeit2) in vorhandene:
                    print(f"⚠️ Kombination {zeit1} - {zeit2} existiert bereits. Skip.")
                    continue

                # Metadata insert
                cur.execute("""
                    INSERT INTO zeitreihe_metadata (startzeit, endzeit, max_value_historic, max_value_upperpi_80_perc)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (zeit1, zeit2, int(max_vergleich), int(max_upper)))
                metadata_id = cur.fetchone()[0]

                for i in range(144):
                    buffer.write(f"{metadata_id}\t{i}\t{round(prediction[i*5], 1)}\t{round(vergleich[i*5], 1)}\t{round(lower[i*5], 1)}\t{round(upper[i*5], 1)}\n")

                print(f"✅ Metadata ID {metadata_id} vorbereitet.")

            buffer.seek(0)

            cur.copy_from(
                buffer,
                'zeitreihe_daten',
                sep='\t',
                columns=('metadata_id', 'minute_value', 'prediction_value', 'historic_value', 'lowerpi_value', 'upperpi_value')
            )

            conn.commit()

            print("✅ Alle Daten mit COPY (psycopg2) eingespielt.")

if __name__ == "__main__":
    iso_date = get_latest_endzeitpunkt_iso(conn_str)
    json_daten = osw_api_extract(stunden_zurueck(iso_date, 11), fast_now(), api_url_template)
    df3 = transform(json_daten)
    results = inference(df3, stunden_zurueck(iso_date, 11))
    load_in_db2(conn_str, results)
