import pandas as pd
import matplotlib.pyplot as plt

# CSV laden und Zeitreihe erstellen
df = pd.read_csv(r'C:\Users\hamadi.alkanaan.HERFORD\Desktop\mein projekt\git\hs\herford_weather.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Zeitraum 1.-3. Juni 2022 filtern
june = df.loc['2022-06-01':'2022-06-03', 'temperature_2m (°C)']

# Glättung mit rolling (6-Stunden-Fenster)
june_smooth = june.rolling(window=6).mean()

# Zwei Plots nebeneinander
fig, axes = plt.subplots(1, 2, figsize=(14,5))

june.plot(ax=axes[0], title='Ohne Glättung')
axes[0].set_ylabel('Temperatur (°C)')

june_smooth.plot(ax=axes[1], title='Mit Glättung (rolling 6h)')
axes[1].set_ylabel('Temperatur (°C)')

plt.tight_layout()
plt.show()
