import pandas as pd
import matplotlib.pyplot as plt

# CSV laden und Zeitreihe erstellen
df = pd.read_csv(r'C:\Users\hamadi.alkanaan.HERFORD\Desktop\mein projekt\git\hs\herford_weather.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# a) Nur 1. August 2022 filtern
aug1 = df.loc['2022-08-01', 'temperature_2m (°C)'].copy()
aug1.plot(figsize=(10,4), title='1. August 2022 - Original')
plt.ylabel('Temperatur (°C)')
plt.show()

# b) Wert um 18:00 entfernen
aug1_missing = aug1.drop(pd.Timestamp('2022-08-01 18:00:00'))
aug1_missing.plot(figsize=(10,4), title='1. August 2022 - ohne 18:00')
plt.ylabel('Temperatur (°C)')
plt.show()

# c) Fehlenden Wert mit globalem Mittelwert auffüllen
aug1_mean = aug1_missing.reindex(aug1.index)
global_mean = aug1_mean.mean()
aug1_filled_mean = aug1_mean.fillna(global_mean)
aug1_filled_mean.plot(figsize=(10,4), title='Aufgefüllt mit Mittelwert')
plt.ylabel('Temperatur (°C)')
plt.show()

# d) Fehlenden Wert mit Lag 1 (vorheriger Wert) auffüllen
aug1_filled_lag = aug1_mean.ffill()
aug1_filled_lag.plot(figsize=(10,4), title='Aufgefüllt mit Lag 1')
plt.ylabel('Temperatur (°C)')
plt.show()

# Erklärung: Mittelwert springt zur Tagesmitte (~19.5°C),
# Lag 1 nimmt den Wert von 17:00 Uhr, was natürlicher aussieht.
