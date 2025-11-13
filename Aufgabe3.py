import pandas as pd

# ---------------------------------------------
# Aufgabe 3a: Arbeitstage (Business Days)
# ---------------------------------------------

# Alle Arbeitstage zwischen 24.12.2025 und 06.01.2026
dates = pd.date_range("2025-12-24", "2026-01-06", freq="B")
print(dates)

# Anzahl der Arbeitstage ausgeben
print("Es gibt", len(dates),
      "normale Wochentage/reguläre Arbeitstage in den weihnachtensferien 2025/2026.")

print("\n\n")

# ---------------------------------------------
# Aufgabe 3b: Sonntage, die auf den 1. fallen
# ---------------------------------------------

# Alle Tage zwischen 24.12.2025 und 06.01.2027
all_dates = pd.date_range("2025-12-24", "2027-01-06")

# Filter: Sonntag (dayofweek == 6) UND Tag == 1
# .to_series().dt -> vermeidet rote Fehlerlinien in VS Code
sundays_first = all_dates[
    (all_dates.to_series().dt.dayofweek == 6) &
    (all_dates.to_series().dt.day == 1)
]

# Gefundene Sonntage anzeigen
print(sundays_first)

# Anzahl anzeigen
print("Vom 24.12.2025 bis zum 6.01.2027 gibt es", len(sundays_first),
      "Sonntage, die auf den 1. des Monats fallen.")
