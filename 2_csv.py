import json
import csv

with open("sensor_data.json", "r") as f:
    data = json.load(f)  # data ist eine Liste von Dictionaries
if isinstance(data, list):
    measurements = data
else:
    print("Fehler: JSON-Daten haben nicht das erwartete Format")
    measurements = []
with open("sensor_data.csv", "w", newline="") as csvfile:
    fieldnames = ['begin', 'v']  # Die Spaltennamen aus der json
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(measurements)

print(" CSV-Datei wurde erfolgreich gespeichert!")
