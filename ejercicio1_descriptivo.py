import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AB_NYC_2019.csv")

"""Análisis descriptivo del dataset de Airbnb en NYC"""

print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])
print("Dimensión:", df.shape)

"""Memoria usada por el DataFrame (en MB) y tipos de datos de cada columna."""

print("\nMemoria usada:")
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

print("\nTipos de datos:")
print(df.dtypes)

"""Número de valores nulos por columna y porcentaje respecto al total."""
nulos = df.isna().sum()
porcentaje_nulos = (nulos / len(df)) * 100

resumen_nulos = pd.DataFrame({
    "nulos": nulos,
    "porcentaje": porcentaje_nulos
})

print(resumen_nulos)

"""Los he dejado tal sin tocarlos porque los nulos aparecen en columnas de variables 
categóricas que no se repiten casi nada, o en variables numéricas que no me parecen 
tan interesantes para el análisis descriptivo."""

########
"""Seleccionamos unicamente las variables númericas y hacemos el análisis
descriptivo de ellas y las exportamos a un csv que aparece en el output"""

df_numerico = df.select_dtypes(include=['int64', 'float64'])
descripcion = df_numerico.describe()
descripcion.to_csv("output/ej1_descriptivo.csv")

"""Calculamos el rango intercuartílico (IQR) para la variable 'price' y también su asimetría y curtosis."""
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

skewness = df['price'].skew()
curtosis = df['price'].kurtosis()

print("Asimetría:", skewness)
print("Curtosis:", curtosis)

"""Aqui están hechos los histogramas de estas 4 variables numéricas 1 a 1 para poder
ajustar los limites de los ejes x y el número de barras con el objetivo de poder apreciar
cada uno con la mayor claridad posible."""
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

##precio
price = df['price'].dropna()
axes[0,0].hist(price, bins=2000)
axes[0,0].set_xlim(0, 500)
axes[0,0].set_title("price")
axes[0,0].set_xlabel("Precio")
axes[0,0].set_ylabel("Frecuencia")

##mínimo número de noches
nights = df['minimum_nights'].dropna()
axes[0,1].hist(nights, bins=2000)
axes[0,1].set_xlim(0, 35)
axes[0,1].set_title("minimum_nights")
axes[0,1].set_xlabel("Noches mínimas")
axes[0,1].set_ylabel("Frecuencia")

## número de reviews
reviews = df['number_of_reviews'].dropna()
axes[1,0].hist(reviews, bins=1000)
axes[1,0].set_xlim(0, 50)
axes[1,0].set_title("number_of_reviews")
axes[1,0].set_xlabel("Número de reviews")
axes[1,0].set_ylabel("Frecuencia")

## número de alojamientos por host
hosts = df['calculated_host_listings_count'].dropna()
axes[1,1].hist(hosts, bins=1000)
axes[1,1].set_xlim(0, 10)  # suele concentrarse aquí
axes[1,1].set_title("calculated_host_listings_count")
axes[1,1].set_xlabel("Nº alojamientos por host")
axes[1,1].set_ylabel("Frecuencia")

plt.tight_layout()
plt.savefig("output/ej1_histogramas.png", dpi=150)

"""Ahora hago los boxplots para comparar el precio entre las diferentes categorías 
de 'neighbourhood_group' y 'room_type' para poder ver si hay diferencias significativas 
entre las categorías y también para detectar posibles outliers en cada una de ellas."""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

## precio por vecindario
sns.boxplot(x='neighbourhood_group', y='price', data=df, ax=axes[0])
axes[0].set_title("Precio por zona")
axes[0].set_xlabel("Zona")
axes[0].set_ylabel("Precio")
axes[0].set_ylim(0, 500)  # limitar por outliers
axes[0].tick_params(axis='x', rotation=45)

## precio por tipo de habitación
sns.boxplot(x='room_type', y='price', data=df, ax=axes[1])
axes[1].set_title("Precio por tipo de habitación")
axes[1].set_xlabel("Tipo de habitación")
axes[1].set_ylabel("Precio")
axes[1].set_ylim(0, 500)

plt.tight_layout()
plt.savefig("output/ej1_boxplots.png", dpi=150)

####

"""Usamos el metodo del rango intercuartílico (IQR) para detectar outliers
en la variable 'price' y calculamos el número y porcentaje de outliers
respecto al total de datos."""
col = "price"

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]

print("Número de outliers:", len(outliers))
print("Porcentaje:", len(outliers) / len(df) * 100)

"""Y ahora aplicamos la variante IQR x3 para detectar unicamente los outliers más extremos"""

limite_superior_x3 = Q3 + 3 * IQR

outliers_x3 = df[df[col] > limite_superior_x3]

print("Número de outliers superiores (3xIQR):", len(outliers_x3))
print("Porcentaje:", len(outliers_x3) / len(df) * 100)

"""Hemos usado en ambos casos el método IQR porque el Z-score solo es para 
distribuciones normales y nuestra distribución de precios no es normal.
"""

####
"""Hago esto porque como en el host_name unicamente aparece el nombre del host y no su apellido,
puede haber hosts distintos con distinto host_id pero con el mismo nombre, aunque 
con esto comprobamos que cada host_id corresponde a un unico host_name"""
print(df.groupby('host_id')['host_name'].nunique().sort_values(ascending=False))

"""Y ahora vemos la frecuencia absoluta y relativa de cuantos alojamientos tiene cada persona,
cuantos alojamientos hay en cada zona, en cada barrio y cuantos hay según el tipo de habitación."""

columnas = ['host_name', 'neighbourhood_group', 'neighbourhood', 'room_type']

for col in columnas:
    print(f"\n--- {col.upper()} ---")
    
    frecuencias = pd.DataFrame({
        "Frecuencia_absoluta": df[col].value_counts(),
        "Frecuencia_relativa_%": df[col].value_counts(normalize=True) * 100
    })
    
    print(frecuencias)

"""Y ahora hacemos los gráficos de barras con cada uno de ellos"""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

## zona
df['neighbourhood_group'].value_counts().plot(
    kind='bar', ax=axes[0,0]
)
axes[0,0].set_title("neighbourhood_group")
axes[0,0].tick_params(axis='x', rotation=45)

## tipo de habitación
df['room_type'].value_counts().plot(
    kind='bar', ax=axes[0,1]
)
axes[0,1].set_title("room_type")
axes[0,1].tick_params(axis='x', rotation=45)

## vecindarios(unicamente los 10 más frecuentes porque hay muchos)
df['neighbourhood'].value_counts().head(10).plot(
    kind='bar', ax=axes[1,0]
)
axes[1,0].set_title("Top 10 neighbourhood")
axes[1,0].tick_params(axis='x', rotation=45)

## los hosts(unicamente los 10 más frecuentes porque hay muchos)
df['host_name'].value_counts().head(10).plot(
    kind='bar', ax=axes[1,1]
)
axes[1,1].set_title("Top 10 host_name")
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("output/ej1_categoricas.png", dpi=150)

"""Y vemos en el output de categóricas que tanto en zonas como en tipo de habitación
hay categorías que dominan por encima del resto."""

####

# Matriz de correlación
corr = df_numerico.corr(method='pearson')

# Crear heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)

plt.title("Matriz de correlación (Pearson)")
plt.tight_layout()

# Guardar con el nombre que quieres
plt.savefig("output/ej1_heatmap_correlacion.png", dpi=150)


   