import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


df = pd.read_csv("AB_NYC_2019.csv")

"""Quitamos estas columnas ya que no nos útiles a la hora de hacer la regresión,
ya que son variables que no se repiten o que considero que no aportan una información relevante."""

df_modelo = df.drop(columns=[
    'id',
    'name',
    'host_name',
    'last_review'
])

"""Ahora convertimos nuestras variables categóricas en numéricas para poder 
hacer la regresión."""

df_modelo = pd.get_dummies(df_modelo, columns=[
    'neighbourhood_group',
    'neighbourhood',
    'room_type'
], drop_first=True)


"""Ahora estandarizamos los valore de nuestras distribuciónes numéricas 
para que todas tengan la misma escala y no haya ninguna que domine sobre las demás
a la hora de hacer la regresión."""
scaler = StandardScaler()
cols_numericas = df_modelo.select_dtypes(include=['int64', 'float64']).columns
df_modelo[cols_numericas] = scaler.fit_transform(df_modelo[cols_numericas])

####
"""Dividimos los datos en dos, el 80% para entrenamiento y el 20% para test. Eliminando antes los valores nulos."""
df_modelo = df_modelo.dropna()

y = df_modelo['price']

"""X son el resto de columnas excepto la objetivo que es el precio"""
X = df_modelo.drop(columns=['price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
####
"""Creamos y entrenamos nuestro modelo con nuestros datos."""

modelo = LinearRegression()
modelo.fit(X_train, y_train)

####

"""Las predicciones de nuestro modelo."""
y_pred = modelo.predict(X_test)

"""Definimos nuestras metricas de evaluación sobre nuesro modelo"""
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

with open("output/ej2_metricas_regresion.txt", "w") as f:
    f.write("MÉTRICAS DEL MODELO DE REGRESIÓN\n")
    f.write("================================\n\n")
    
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2: {r2:.4f}\n")


####

"""Cálculo de residuos y creación del gráfico de residuos para evaluar la calidad
de nuestro modelo de regresión."""
residuos = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuos,s=8, alpha=0.3)

"""Añadimos una linea horizontal para que sea más fácil visualizar la distancia de
nuestros residuos a 0, el punto óptimo para tener el mejor modelo"""

plt.axhline(0)

plt.title("Gráfico de residuos")
plt.xlabel("Valores predichos")
plt.ylabel("Residuos")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("output/ej2_residuos.png", dpi=150)



