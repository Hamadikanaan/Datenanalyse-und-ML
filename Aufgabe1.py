import pandas as pd
import matplotlib.pyplot as plt

# CSV laden
df = pd.read_csv(r'C:\Users\hamadi.alkanaan.HERFORD\Desktop\mein projekt\git\hs\herford_weather.csv')

# a) Zeitstempel in DateTime umwandeln und als Index setzen
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df.index.name = 'Date Time'

# Erste 5 Zeilen anzeigen
print(df.head())

# Liniendiagramm für temperature_2m
df['temperature_2m (°C)'].plot(figsize=(12,5))
plt.ylabel('Temperatur (°C)')
plt.title('Temperatur in 2m Höhe')
plt.show()

# b) Nur 2022 filtern und tägliche Mittelwerte
df_2022 = df.loc['2022-01-01':'2022-12-31']
df_2022_daily = df_2022.resample('D').mean()

print(df_2022_daily.head())

# Zwei Plots: stündlich und täglich für 2022
fig, axes = plt.subplots(1, 2, figsize=(14,5))

df_2022['temperature_2m (°C)'].plot(ax=axes[0], title='Stündliche Werte 2022')
axes[0].set_ylabel('Temperatur (°C)')

df_2022_daily['temperature_2m (°C)'].plot(ax=axes[1], title='Tägliche Mittelwerte 2022')
axes[1].set_ylabel('Temperatur (°C)')

plt.tight_layout()
plt.show()
