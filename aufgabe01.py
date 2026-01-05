import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Daten laden und vorbereiten
df = pd.read_csv('rawdata_luftqualitaet.csv')

features = ['humidity_inside', 'temperature_inside', 'co2_inside', 
            'temperature_heater', 'temperature_wall_inside']
X = df[features]    # Eingabemerkmale
y = df['state_air_quality'] # Zielvariable

scaler = StandardScaler()       
X_scaled = scaler.fit_transform(X)  # fit und transformiere macht Standardisierung

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)    

# ===================== AUFGABE 1 =====================
model1 = Sequential([
    Dense(60, activation='relu', input_shape=(5,)),
    Dense(60, activation='relu'),
    Dense(3, activation='softmax')
])
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.summary()

history1 = model1.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=0)

plt.figure()
plt.plot(history1.history['loss'], label='train loss')
plt.plot(history1.history['val_loss'], label='test loss')
plt.xlabel('epochs')
plt.ylabel('loss (sparse cross entropy)')
plt.legend()
plt.title('Aufgabe 1: Overfitting')
plt.show()

# ===================== AUFGABE 2 =====================
model2 = Sequential([
    Dense(60, activation='relu', input_shape=(5,)),
    Dense(60, activation='relu'),
    Dense(3, activation='softmax')
])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history2 = model2.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), 
                      callbacks=[early_stop], verbose=0)

plt.figure()
plt.plot(history2.history['loss'], label='train loss')
plt.plot(history2.history['val_loss'], label='test loss')
plt.xlabel('epochs')
plt.ylabel('loss (sparse cross entropy)')
plt.legend()
plt.title('Aufgabe 2: EarlyStopping')
plt.show()

# ===================== AUFGABE 3 =====================
model3 = Sequential([
    Dense(60, activation='relu', input_shape=(5,), kernel_regularizer=l2(0.01)),
    Dense(60, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(3, activation='softmax')
])
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history3 = model3.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=0)

plt.figure()
plt.plot(history3.history['loss'], label='train loss')
plt.plot(history3.history['val_loss'], label='test loss')
plt.xlabel('epochs')
plt.ylabel('loss (sparse cross entropy)')
plt.legend()
plt.title('Aufgabe 3: L2-Regularisierung')
plt.show()