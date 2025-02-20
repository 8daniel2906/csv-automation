import requests
import json
from datetime import datetime, timedelta

# Aktuelles Datum und Uhrzeit abrufen
jetzt = datetime.now()

# Erster Zeitpunkt: Morgen um 00:00 Uhr
morgen = (jetzt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

# Zweiter Zeitpunkt: Vor 7 Tagen um 00:00 Uhr
vor_sechs_tagen = (jetzt - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

# API-Link mit Platzhaltern
api_url_template = "https://api.opensensorweb.de/v1/organizations/pikobytes/networks/bafg/devices/5952025/sensors/w/measurements/raw?start={start}%2B01:00&end={end}%2B01:00&interpolator=LINEAR"

# Link mit den berechneten Datumswerten ersetzen
api_url = api_url_template.format(start=vor_sechs_tagen, end=morgen)

# Ergebnisse ausgeben
print("API-Link:", api_url)

# API URL (mit deinem spezifischen Parameter)
url = "https://api.opensensorweb.de/v1/organizations/pikobytes/networks/bafg/devices/5952025/sensors/w/measurements/raw?start=2025-02-12T00:00:00%2B01:00&end=2025-02-19T22:59:59%2B01:00&interpolator=LINEAR"

# Anfrage an die API
response = requests.get(url)

# Überprüfen, ob die Anfrage erfolgreich war
if response.status_code == 200:
    # Die Antwort ist im JSON-Format, also können wir sie speichern oder weiterverarbeiten
    data = response.json()

    # Optional: Speichern der Antwort in einer JSON-Datei
    with open('sensor_data.json', 'w') as f:
        json.dump(data, f, indent=4)

    print("Daten erfolgreich gespeichert!")
else:
    print(f"Fehler beim Abrufen der Daten. Statuscode: {response.status_code}")
