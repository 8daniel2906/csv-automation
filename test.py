from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import shutil

# Pfad zum ChromeDriver (ändern Sie diesen Pfad entsprechend Ihres Systems)

#driver_path = 'D:\chromedriver-win64\chromedriver-win64\chromedriver.exe'  # z.B. 'D:/chromedriver.exe'
driver_path = '/usr/local/bin/'




# Verwenden Sie die Service-Klasse, um den Pfad zum WebDriver anzugeben
service = Service(driver_path)

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Headless-Modus für GitHub Actions
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")





# Initialisieren des WebDrivers
driver = webdriver.Chrome(service=service, options=options)



# Öffnen der Website
driver.get("https://www.opensensorweb.de/de/data/?c=9.978306%2C53.468213&sid=pikobytes%24bafg%245952025%24w&v=sidebar&te=2025-02-11T23%3A00%3A00.000Z%2C2025-02-19T22%3A59%3A59.999Z&z=13.88")

time.sleep(2)

try:
    # Warten, bis das Cookie-Banner sichtbar ist, und auf "Erlauben" klicken
    cookie_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Erlauben')]"))  # Verwenden Sie eine der Klassen
    )
    cookie_button.click()
    print("Cookie-Banner wurde akzeptiert.")
except Exception as e:
    print("Cookie-Banner nicht gefunden oder konnte nicht geklickt werden:", e)

time.sleep(2)

try:
    # Warten, bis der Download-Button sichtbar ist, und darauf klicken
    download_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.MuiButtonBase-root > svg[data-testid='DownloadIcon']"))
    )
    download_button.click()
    print("Kleiner-Download-Icon-Button wurde geklickt.")
except Exception as e:
    print("Kleiner-Download-Icon-Button nicht gefunden oder konnte nicht geklickt werden:", e)


time.sleep(2)


try:
    # Warten, bis das Cookie-Banner sichtbar ist, und auf "Erlauben" klicken
    cookie_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div[3]/div/div[2]/div/div[2]/div/div"))  # Verwenden Sie eine der Klassen
    )
    cookie_button.click()
    print("Auflösung wurde ausgerollt.")
except Exception as e:
    print("Auflösung wurde nicht ausgerollt oder konnte nicht geklickt werden:", e)

time.sleep(2)

try:
    # Warten, bis das Cookie-Banner sichtbar ist, und auf "Erlauben" klicken
    cookie_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/div[4]/div[3]/ul/li[13]"))  # Verwenden Sie eine der Klassen
    )
    cookie_button.click()
    print("Richtige Auflösung wurde akzeptiert.")
except Exception as e:
    print("Richtige Auflösung wurde nicht gefunden oder konnte nicht geklickt werden:", e)

time.sleep(2)
try:
    # Warten, bis das Cookie-Banner sichtbar ist, und auf "Erlauben" klicken
    cookie_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div[3]/div/footer/div"))  # Verwenden Sie eine der Klassen
    )
    cookie_button.click()
    print("Dowload Button (rot) wurde akzeptiert.")
except Exception as e:
    print("Dowload Button (rot) nicht gefunden oder konnte nicht geklickt werden:", e)

time.sleep(5) #warte zeit damit es fertig downloaden kann

# Verschiebe die heruntergeladene Datei ins Repository
download_folder = os.path.expanduser("~/Downloads")
destination_folder = os.path.expanduser(".")

for file in os.listdir(download_folder):
    if file.endswith(".zip"):  # Passe das Dateiformat an
        shutil.move(os.path.join(download_folder, file), os.path.join(destination_folder, file))
        print(f"Datei {file} wurde ins Repository verschoben.")

#input()
# Schließen des Browsers
driver.quit()