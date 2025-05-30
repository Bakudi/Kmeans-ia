# 🧠 PokeMeans - Clustering de Pokémon

# --- 1. Importación de librerías ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 2. Carga de datos ---
df = pd.read_csv('Pokemon.csv')
print("Columnas del dataset:")
print(df.columns)

# --- 3. Selección de estadísticas relevantes ---
stats_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
df_stats = df[stats_cols]

# --- 4. Normalización de datos ---
scaler = StandardScaler()
scaled_stats = scaler.fit_transform(df_stats)

# --- 5. Aplicación de K-Means ---
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_stats)

# --- 6. Reducción de dimensionalidad con PCA ---
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_stats)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

# --- 7. Visualización 2D de clusters ---
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
plt.title("Clusters de Pokémon (PCA + K-Means)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.show()

# --- 8. Análisis estadístico por cluster ---
cluster_summary = df.groupby('Cluster')[stats_cols].mean().round(1)
print("Resumen estadístico por cluster:")
print(cluster_summary)

# --- 9. Asignar nombres creativos a los clusters ---
cluster_names = {
    0: 'Tanques Estratégicos',
    1: 'Cañones de Cristal',
    2: 'Equilibrados',
    3: 'Velocistas Frágiles'
}
df['ClusterName'] = df['Cluster'].map(cluster_names)

# --- 10. Heatmap comparativo de stats ---
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_summary, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Promedio'})
plt.title("Estadísticas promedio por Cluster")
plt.show()

# --- 11. Análisis detallado de cada cluster ---
for i, row in cluster_summary.iterrows():
    print(f"\nCluster {i} - {cluster_names[i]}")
    print(row.sort_values(ascending=False))

# --- 12. Visualización adicional: distribución de tipos por cluster ---
plt.figure(figsize=(12, 6))
cluster_types = df.groupby('Cluster')['Type 1'].value_counts().unstack().fillna(0)
cluster_types.plot(kind='bar', stacked=True, colormap='Set3', figsize=(12, 6))
plt.title("Distribución de Tipos por Cluster")
plt.xlabel("Cluster")
plt.ylabel("Cantidad de Pokémon")
plt.legend(title='Tipo')
plt.show()

# --- 13. ¿Dónde están los Pokémon legendarios? ---
legendary_distribution = df.groupby(['Cluster', 'Legendary']).size().unstack(fill_value=0)
print("\nDistribución de Pokémon legendarios por cluster:")
print(legendary_distribution)

# --- 14. Cluster personalizado: Pokémon con ADN de Jefe Final ---

# Definimos umbrales para ser considerado "jefe final"
# Alto en HP, Attack, Defense o Sp. Atk (top 25%)
# Bajo en Speed (bottom 25%)
high_threshold = df[stats_cols].quantile(0.75)
low_speed_threshold = df['Speed'].quantile(0.25)

# Condición compuesta
jefes_df = df[
    ((df['HP'] >= high_threshold['HP']) |
     (df['Attack'] >= high_threshold['Attack']) |
     (df['Defense'] >= high_threshold['Defense']) |
     (df['Sp. Atk'] >= high_threshold['Sp. Atk'])) &
    (df['Speed'] <= low_speed_threshold)
].copy()

# Nombre del cluster
jefes_df['CustomCluster'] = 'ADN de Jefe Final'

print(f"\n🛡️ Pokémon con ADN de Jefe Final ({len(jefes_df)} encontrados):")
print(jefes_df[['Name', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Speed']].sort_values(by='HP', ascending=False))

# Visualización en PCA con el resto del dataset
df['CustomCluster'] = 'Normal'
df.loc[jefes_df.index, 'CustomCluster'] = 'ADN de Jefe Final'

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='CustomCluster', palette={'Normal': 'gray', 'ADN de Jefe Final': 'darkred'}, s=80)
plt.title("Pokémon con ADN de Jefe Final (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Tipo de Pokémon')
plt.show()

# Estadísticas promedio de los jefes
print("\n📊 Estadísticas promedio del cluster 'ADN de Jefe Final':")
print(jefes_df[stats_cols].mean().round(1))

