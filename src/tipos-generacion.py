import pandas as pd

# Cargar los datos
df = pd.read_csv("Pokemon.csv")

# Filtrar solo Pokémon con doble tipo
df = df.dropna(subset=["Type 2"])

# Agrupar por combinación exacta de tipos
grouped = df.groupby(["Type 1", "Type 2"])

# Crear lista para guardar resultados
clusters = []
cluster_id = 0

for (type1, type2), group in grouped:
    if group["Generation"].nunique() > 1:
        clusters.append({
            "Cluster": cluster_id,
            "Tipo 1": type1,
            "Tipo 2": type2,
            "Generaciones": sorted(group["Generation"].unique()),
            "Pokémon": group["Name"].tolist()
        })
        cluster_id += 1

# Mostrar resultados
for cluster in clusters:
    print(f"\n🔷 Cluster {cluster['Cluster']} — Tipos: {cluster['Tipo 1']}/{cluster['Tipo 2']} (Gen: {cluster['Generaciones']})")
    for name in cluster["Pokémon"]:
        print(f" - {name}")


