import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('herford_weather.csv')
df.columns = df.columns.str.replace('Â', '')
target = 'dewpoint_2m (°C)'
features = [col for col in df.columns if col not in [target, 'time']]

# Aufgabe a
print("Aufgabe a:")
korrelation = df[features].corrwith(df[target]).sort_values(ascending=False)
print("Korrelation mit Taupunkt:")
print(korrelation)

plt.figure(figsize=(12, 8))
korrelation.plot(kind='barh')
plt.xlabel('Korrelation')
plt.title('Korrelation der Merkmale mit Taupunkt')
plt.tight_layout()
plt.savefig('korrelation_taupunkt.png')
plt.close()

ausgewaehlte_merkmale = ['temperature_2m (°C)', 'apparent_temperature (°C)', 
                         'soil_temperature_0_to_7cm (°C)', 'relativehumidity_2m (%)']
print("\nAusgewählte Merkmale:", ausgewaehlte_merkmale)
print("Begründung: Diese Merkmale haben die höchste Korrelation mit dem Taupunkt.")

# Aufgabe b
print("\nAufgabe b:")
X = df[ausgewaehlte_merkmale].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(1, input_dim=len(ausgewaehlte_merkmale)))
model.compile(optimizer='adam', loss='mse')
print("Modell erstellt. Optimizer: Adam, Loss: MSE")

# Aufgabe c
print("\nAufgabe c:")
X_train_small = X_train_scaled[:1000]
y_train_small = y_train[:1000]

history = model.fit(X_train_small, y_train_small, 
                    epochs=100, batch_size=32, 
                    validation_split=0.2, verbose=0)
print("Modell trainiert. Epochs: 100, Batch Size: 32")

# Aufgabe d
print("\nAufgabe d:")
y_pred = model.predict(X_test_scaled, verbose=0)
r2 = r2_score(y_test, y_pred)
print(f"R-Quadrat: {r2:.4f}")

# Aufgabe e
print("\nAufgabe e:")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Lernkurve - Taupunkt Vorhersage')
plt.legend()
plt.savefig('lernkurve_taupunkt.png')
plt.close()
print("Lernkurve gespeichert als 'lernkurve_taupunkt.png'")

# Aufgabe f
print("\nAufgabe f:")
target_snow = 'snowfall (cm)'
korrelation_snow = df[features].corrwith(df[target_snow]).sort_values(ascending=False)
print("Korrelation mit Schneefall:")
print(korrelation_snow)

plt.figure(figsize=(12, 8))
korrelation_snow.plot(kind='barh')
plt.xlabel('Korrelation')
plt.title('Korrelation der Merkmale mit Schneefall')
plt.tight_layout()
plt.savefig('korrelation_schneefall.png')
plt.close()

merkmale_snow = ['precipitation (mm)', 'temperature_2m (°C)', 
                 'apparent_temperature (°C)', 'weathercode (wmo code)']

X_snow = df[merkmale_snow].values
y_snow = df[target_snow].values
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_snow, y_snow, test_size=0.2, random_state=42)

scaler_snow = StandardScaler()
X_train_s_scaled = scaler_snow.fit_transform(X_train_s)
X_test_s_scaled = scaler_snow.transform(X_test_s)

model_snow = Sequential()
model_snow.add(Dense(1, input_dim=len(merkmale_snow)))
model_snow.compile(optimizer='adam', loss='mse')

X_train_s_small = X_train_s_scaled[:1000]
y_train_s_small = y_train_s[:1000]

history_snow = model_snow.fit(X_train_s_small, y_train_s_small,
                               epochs=100, batch_size=32,
                               validation_split=0.2, verbose=0)

y_pred_snow = model_snow.predict(X_test_s_scaled, verbose=0)
r2_snow = r2_score(y_test_s, y_pred_snow)
print(f"\nR-Quadrat Schneefall: {r2_snow:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(history_snow.history['loss'], label='Training Loss')
plt.plot(history_snow.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Lernkurve - Schneefall Vorhersage')
plt.legend()
plt.savefig('lernkurve_schneefall.png')
plt.close()

print("\nErklärung: R² für Schneefall ist niedrig, weil Schneefall selten ist (viele Nullwerte).")
print("Lineare Regression ist nicht gut für solche Daten.")