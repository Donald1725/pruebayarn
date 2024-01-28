import pandas as pd
import numpy as np
import skfuzzy as fuzz
import sys
from sklearn.preprocessing import StandardScaler

for ruta in sys.stdin:
    # Definir la URL del archivo CSV 

    url = ruta.strip()

    # Leer el archivo CSV y almacenarlo en un DataFrame de pandas
    #df = pd.read_csv(url, sep=';', error_bad_lines=False)
    df = pd.read_csv(url, sep=';', on_bad_lines='skip')

    # Convertir las columnas a numéricas, reemplazando las comas por puntos y convirtiendo los errores en NaN
    df['O3'] = pd.to_numeric(df['O3'].str.replace(',', '.'), errors='coerce')
    df['CO'] = pd.to_numeric(df['CO'].str.replace(',', '.'), errors='coerce')
    df['NO2'] = pd.to_numeric(df['NO2'].str.replace(',', '.'), errors='coerce')
    df['SO2'] = pd.to_numeric(df['SO2'].str.replace(',', '.'), errors='coerce')
    df['PM2_5'] = pd.to_numeric(df['PM2_5'].str.replace(',', '.'), errors='coerce')

    # Seleccionar las columnas relevantes y convertirlas en arrays numpy
    data_o3 = df[['O3']].values
    data_co = df[['CO']].values
    data_no2 = df[['NO2']].values
    data_so2 = df[['SO2']].values
    data_pm25 = df[['PM2_5']].values

    # Crear un objeto StandardScaler para escalar los datos
    scaler = StandardScaler()

    # Escalar los datos para cada columna
    data_o3_scaled = scaler.fit_transform(data_o3)
    data_co_scaled = scaler.fit_transform(data_co)
    data_no2_scaled = scaler.fit_transform(data_no2)
    data_so2_scaled = scaler.fit_transform(data_so2)
    data_pm25_scaled = scaler.fit_transform(data_pm25)

    # Función para realizar Fuzzy C-means
    def fuzzy_c_means(data, num_clusters):
        # Aplicar el algoritmo Fuzzy C-means a los datos
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, num_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        # Determinar la pertenencia a los clusters
        cluster_membership = np.argmax(u, axis=0)
        return cluster_membership

    # Definir el número de clusters para cada columna
    num_clusters_o3 = 3
    num_clusters_co = 3
    num_clusters_no2 = 3
    num_clusters_so2 = 3
    num_clusters_pm25 = 3

    # Aplicar Fuzzy C-means a cada columna
    clusters_o3 = fuzzy_c_means(data_o3_scaled, num_clusters_o3)
    clusters_co = fuzzy_c_means(data_co_scaled, num_clusters_co)
    clusters_no2 = fuzzy_c_means(data_no2_scaled, num_clusters_no2)
    clusters_so2 = fuzzy_c_means(data_so2_scaled, num_clusters_so2)
    clusters_pm25 = fuzzy_c_means(data_pm25_scaled, num_clusters_pm25)

    # Agregar los resultados al DataFrame original
    df['Cluster_O3'] = clusters_o3
    df['Cluster_CO'] = clusters_co
    df['Cluster_NO2'] = clusters_no2
    df['Cluster_SO2'] = clusters_so2
    df['Cluster_PM2_5'] = clusters_pm25

    # Visualizar los resultados
    print(df.head())

    # Guardar el DataFrame 'df' en un archivo de Excel
    df.to_excel('datafinal_clusters.xlsx', index=False)