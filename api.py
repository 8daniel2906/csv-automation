import requests
import json
from datetime import datetime, timedelta

jetzt = datetime.now() # in format beispielsweise: 2025-03-06T00:00:00
# Erster Zeitpunkt: Morgen um 00:00 Uhr
morgen = (jetzt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat() #isoformat zb. 2007-08-31T16:47+00:00

# Zweiter Zeitpunkt: Vor 8 Tagen um 10:00 Uhr, weil letzte 14 stunden
vor_acht_tagen = (jetzt - timedelta(days=8)).replace(hour=10, minute=0, second=0, microsecond=0).isoformat()

# link mit Platzhaltern
#api_url_template = "https://api.opensensorweb.de/v1/organizations/pikobytes/networks/bafg/devices/5952025/sensors/w/measurements/raw?start={start}%2B01:00&end={end}%2B01:00&interpolator=LINEAR"
api_url_template = "https://api.opensensorweb.de/v1/organizations/pikobytes/networks/bafg/devices/5952020/sensors/w/measurements/raw?start={start}%2B01:00&end={end}%2B01:00&interpolator=LINEAR"

api_url = api_url_template.format(start=vor_acht_tagen, end=morgen)
url = api_url
response = requests.get(url) #anfrage an die api wird in response gesaved

if response.status_code == 200: #prüft ob die anfrage zur api funktioniert hat
    data = response.json() #der json-inhalt der anfrage wird gespeichert

    with open('sensor_data.json', 'w') as f: #datei landet für 2_csv-py im repo
        json.dump(data, f, indent=4)
    print("Daten erfolgreich gespeichert!")
else:
    print(f"Fehler beim Abrufen der Daten. Statuscode: {response.status_code}")
