import pandas as pd

rows = []  # Liste für (Land, Bevölkerung)

with open(r"C:\Users\Hamadi\OneDrive\Desktop\IFM5\ML\aufgabe3\countries_population.csv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()      # Zeile reinigen
        if not line:
            continue             # Leere Zeilen überspringen

        parts = line.rsplit(" ", 1)  # Von hinten in Land + Zahl trennen

        country = parts[0].strip().strip("'")      # Landname ohne '
        population = parts[1].replace(",", "").strip()  # Kommas entfernen

        rows.append((country, int(population)))    # Tupel speichern

df = pd.DataFrame(rows, columns=["Country", "Population"])  # DataFrame bauen

print(df.head(5).to_string(index=False))  # Erste 5 Zeilen ohne Index anzeigen
