import json
import csv

# Lade die JSON-Daten aus der Datei
with open("sensor_data.json", "r") as f:
    data = json.load(f)  # data ist eine Liste von Dictionaries

# Prüfen, ob die Daten eine Liste sind
if isinstance(data, list):
    measurements = data  # Direkt die Liste verwenden
else:
    print("Fehler: JSON-Daten haben nicht das erwartete Format")
    measurements = []

# CSV-Datei schreiben
with open("sensor_data.csv", "w", newline="") as csvfile:
    fieldnames = ['begin', 'v']  # Die Spaltennamen aus der JSON-Struktur
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # Schreibe die Kopfzeile
    writer.writerows(measurements)  # Schreibe die Messwerte in die Datei

print("✅ CSV-Datei wurde erfolgreich gespeichert!")
