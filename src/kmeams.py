# PokeMeans - Clustering de Pokémon
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar datos
df = pd.read_csv('Pokemon.csv')

# Verificar columnas relevantes
print(df.columns)

# Selección de atributos numéricos
stats_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
df_stats = df[stats_cols]

# Normalizar datos
scaler = StandardScaler()
scaled_stats = scaler.fit_transform(df_stats)

# Aplicar K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_stats)

# Visualización básica
sns.pairplot(df, vars=stats_cols, hue='Cluster', palette='Set2')
plt.suptitle("Clustering de Pokémon (K-Means)", y=1.02)
plt.show()


