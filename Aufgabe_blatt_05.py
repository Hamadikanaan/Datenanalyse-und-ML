import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Pfad ggf. anpassen
df = pd.read_csv("rawdata_luftqualitaet.csv", sep=",")

print("Erste Zeilen der Datentabelle:")
print(df.head())

print("\nStatistische Kennwerte (min, max, mean, std, count):")
stats = df.describe().T[["min", "max", "mean", "std", "count"]]
print(stats)

# Liniendiagramme der Messgrößen (ohne Zielvariable)
df.drop(columns=["state_air_quality"]).plot(
    subplots=True, figsize=(10, 8), legend=False, title="Zeitverlauf der Messgrößen"
)
plt.tight_layout()
plt.show()

# Heatmap der Korrelationen
plt.figure(figsize=(6, 5))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Korrelationsmatrix")
plt.tight_layout()
plt.show()

# Scattermatrix / Pairplot
sns.pairplot(df, hue="state_air_quality")
plt.suptitle("Scattermatrix der Merkmale", y=1.02)
plt.show()

# -------------------
# b) Train/Test-Split (80/20)
# -------------------
X = df.drop(columns=["state_air_quality"])
y = df["state_air_quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nAnzahl Messwerte insgesamt:", len(df))
print("Trainingsdaten:", len(X_train))
print("Testdaten:", len(X_test))

# -------------------
# c) Normalisierung auf [0, 1] und Kontrolle
# -------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)

print("\nBeispiel der normierten Trainingsdaten:")
print(X_train_scaled_df.head())

# Kontrolle: Boxplot der normierten Merkmale
plt.figure(figsize=(8, 4))
sns.boxplot(data=X_train_scaled_df)
plt.title("Normierte Trainingsdaten (0–1)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------
# d) MLP-Klassifikator trainieren und Prognosen ausgeben
# -------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 50),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# Prognose für Luftqualitäts-Bewertung auf dem Trainingsdatensatz
y_train_pred = mlp.predict(X_train_scaled)
print("\nPrognosen (Trainingsdaten) – erste 20 Werte:")
print(y_train_pred[:20])

print("Trainingsgenauigkeit:", accuracy_score(y_train, y_train_pred))

# Optional: Prognosen für alle Messwerte
y_all_pred = mlp.predict(scaler.transform(X))
print("\nPrognosen für alle Messwerte – erste 20 Werte:")
print(y_all_pred[:20])

# -------------------
# e) Evaluation auf Testdaten
# -------------------
y_test_pred = mlp.predict(X_test_scaled)

print("\nGenauigkeit auf Testdaten:", accuracy_score(y_test, y_test_pred))
print("\nKonfusionsmatrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nDetaillierter Klassifikationsbericht:")
print(classification_report(y_test, y_test_pred))
