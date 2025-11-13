import pandas as pd

df = pd.read_csv(
    r"C:\Users\Hamadi\OneDrive\Desktop\IFM5\ML\aufgabe3\bundeslaender.txt",
    sep=r"\s+"
)

# population = female + male
df["population"] = df["female"] + df["male"]

# density als int
df["density"] = ((df["population"] / df["area"])*1000).astype(int)

# Spalten in gewünschter Reihenfolge
df = df[["land", "area", "female", "male", "population", "density"]]

# mit Index anzeigen
print(df)

############################################
b = df[df["female"] > df["male"]]
print(b)
print("Anzahl:", len(b))

############################################
c = df[df["density"] > 1000]
print(c)
