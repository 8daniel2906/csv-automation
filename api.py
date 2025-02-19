import requests
import json

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
