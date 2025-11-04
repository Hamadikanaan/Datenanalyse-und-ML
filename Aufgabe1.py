import pandas as pd


data = {'Wien': {'country': 'Österreich', 'area': 414.6, 'population': 1805681},
        'Berlin': {'country': 'Deutschland', 'area': 891.85, 'population': 3562166},
        'Zürich': {'country': 'Schweiz', 'area': 87.88, 'population': 378884}}

## transpotieren : erste und zweite index tauschen
df = pd.DataFrame(data).T

# stack macht tauscht spalten und zeilen neben contries ist jz  3 zeile und eine spalte  ohne stack ist das anderes eine zeile und 3 spalte 
s = df.stack()
print("--- Ausgabe 1a ---")
print(s)

##zweite Aufgabe sortierung
s_sorted = s.sort_index()
print("\n--- Ausgabe 1b ---")
print(s_sorted)


print("\n--- Ausgabe 1c ---")
df1= pd.DataFrame(data)
s1= df1.stack()
sort1 = s1.sort_index()
print(sort1)



####### zweite lösung :
s_swapped = s_sorted.swaplevel()
s_c = s_swapped.sort_index()
print(s_c)