"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 4
Análisis y Descomposición de Series Temporales
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio trabajarás con una serie temporal sintética generada con
una semilla fija. Tendrás que:

  1. Visualizar la serie completa.
  2. Descomponerla en sus componentes: Tendencia, Estacionalidad y Residuo.
  3. Analizar cada componente y responder las preguntas del fichero
     Respuestas.md (sección Ejercicio 4).
  4. Evaluar si el ruido (residuo) se ajusta a un ruido ideal (gaussiano
     con media ≈ 0 y varianza constante).

LIBRERÍAS PERMITIDAS
--------------------
  - numpy, pandas
  - matplotlib, seaborn
  - statsmodels   (para seasonal_decompose y adfuller)
  - scipy.stats   (para el test de normalidad del ruido)

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej4_serie_original.png      → Gráfico de la serie completa
  - output/ej4_descomposicion.png      → Los 4 subgráficos de descomposición
  - output/ej4_acf_pacf.png           → Gráfico ACF y PACF del residuo
  - output/ej4_histograma_ruido.png   → Histograma + curva normal del residuo
  - output/ej4_analisis.txt            → Estadísticos numéricos del análisis

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)


# =============================================================================
# GENERACIÓN DE LA SERIE TEMPORAL SINTÉTICA — NO MODIFICAR ESTE BLOQUE
# =============================================================================

def generar_serie_temporal(semilla=42):
    """
    Genera una serie temporal sintética con componentes conocidos.

    La serie tiene:
      - Una tendencia lineal creciente.
      - Estacionalidad anual (periodo 365 días).
      - Ciclos de largo plazo (periodo ~4 años).
      - Ruido gaussiano.

    Parámetros 
    ----------
    semilla : int — Semilla aleatoria para reproducibilidad (NO modificar)

    Retorna
    -------
    serie : pd.Series con índice DatetimeIndex diario (2018-01-01 → 2023-12-31)
    """
    rng = np.random.default_rng(semilla)

    # Índice temporal: 6 años de datos diarios
    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    # --- Componentes ---
    # 1. Tendencia lineal
    tendencia = 0.05 * t + 50

    # 2. Estacionalidad anual (periodo = 365.25 días)
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) \
                   +  6 * np.cos(4 * np.pi * t / 365.25)

    # 3. Ciclo de largo plazo (periodo ~ 4 años = 1461 días)
    ciclo = 8 * np.sin(2 * np.pi * t / 1461)

    # 4. Ruido gaussiano
    ruido = rng.normal(loc=0, scale=3.5, size=n)

    # Serie completa (modelo aditivo)
    valores = tendencia + estacionalidad + ciclo + ruido

    serie = pd.Series(valores, index=fechas, name="valor")
    return serie


# =============================================================================
# TAREA 1 — Visualizar la serie completa
# =============================================================================

def visualizar_serie(serie):
    """
    Genera y guarda un gráfico de la serie temporal completa.

    Salida: output/ej4_serie_original.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal a visualizar

    Pistas
    ------
    - Usa fig, ax = plt.subplots(figsize=(14, 4))
    - Añade título, etiquetas de ejes y una cuadrícula suave
    - Guarda con plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches='tight')
    """
    
    "creamos la grafica y le damos las variables de nuestra serie"
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(serie.index, serie.values, linewidth=1)

    "añadimos el titulo, las etiquetas de los ejes y la cuadrícula"
    ax.set_title("Serie temporal ", fontsize=12)
    ax.set_xlabel("Años")
    ax.set_ylabel("Valores")
    ax.grid(True, linestyle="--", alpha=0.5)

    "La añadimos en el outuput"
    plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches='tight')

    

# =============================================================================
# TAREA 2 — Descomposición de la serie
# =============================================================================

def descomponer_serie(serie):
    """
    Descompone la serie en Tendencia, Estacionalidad y Residuo usando
    statsmodels.tsa.seasonal.seasonal_decompose y guarda el gráfico.

    Salida: output/ej4_descomposicion.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal

    Retorna
    -------
    resultado : DecomposeResult — Objeto con atributos .trend, .seasonal, .resid

    Pistas
    ------
    - from statsmodels.tsa.seasonal import seasonal_decompose
    - Usa model='additive' y period=365
    - resultado.plot() genera los 4 subgráficos automáticamente
    - Guarda la figura con fig.savefig(...)
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    """Descomponemos la serie en sus componentes y como la amplitud de la estacionalidad es constante,
    usamos el modelo aditivo, el periodo de la serie es de 365 días

    """
    resultado = seasonal_decompose(
        serie,
        model='additive',
        period=365)
    
    fig = resultado.plot()
    
    fig.set_size_inches(12, 8)

    fig.suptitle("Descomposición de la serie temporal", fontsize=14)

    fig.savefig("output/ej4_descomposicion.png", dpi=150, bbox_inches='tight')

    return resultado
    



# =============================================================================
# TAREA 3 — Análisis del residuo (ruido)
# =============================================================================

def analizar_residuo(residuo):
    """
    Analiza el componente de residuo para determinar si se parece
    a un ruido ideal (gaussiano, media ≈ 0, varianza constante, sin autocorr.).

    Genera:
      - output/ej4_acf_pacf.png          → ACF y PACF del residuo
      - output/ej4_histograma_ruido.png  → Histograma + curva normal ajustada
      - output/ej4_analisis.txt          → Estadísticos numéricos

    Parámetros
    ----------
    residuo : pd.Series — Componente de residuo de la descomposición

    Pistas para el análisis numérico
    ----------------------------------
    - Media:     residuo.mean()
    - Std:       residuo.std()
    - Asimetría: residuo.skew()       (ruido ideal ≈ 0)
    - Curtosis:  residuo.kurtosis()   (ruido ideal ≈ 0 en exceso)
    - ADF Test para verificar estacionariedad:
        from statsmodels.tsa.stattools import adfuller
        resultado_adf = adfuller(residuo.dropna())
        p_adf = resultado_adf[1]
    - ACF / PACF:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    """
    #Limpia el residuo (elimina NaN al inicio/fin)

    residuo_limpio = residuo.dropna()

    """Calcula los estadísticos numéricos del residuo:"""
    media = residuo_limpio.mean()
    std = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis = residuo_limpio.kurtosis()

    """Test de normalidad de Jarque-Bera para evaluar si el residuo se ajusta a una distribución normal"""

    from scipy.stats import jarque_bera
    stat_jb, p_jb = jarque_bera(residuo_limpio)

    """Test para verificar la estacionariedad del residuo a traves del p-valor"""
    from statsmodels.tsa.stattools import adfuller
    resultado_adf = adfuller(residuo_limpio)
    p_adf = resultado_adf[1]

    """Guardamos el análisis numérico en un archivo de texto"""

    with open("output/ej4_analisis.txt", "w") as f:
        f.write("ANÁLISIS DEL RESIDUO\n")
        f.write("====================\n\n")

        f.write(f"Media: {media:.4f}\n")
        f.write(f"Desviación estándar: {std:.4f}\n")
        f.write(f"Asimetría: {asimetria:.4f}\n")
        f.write(f"Curtosis: {curtosis:.4f}\n\n")

        f.write(f"Jarque_Bera_stat: {stat_jb:.4f}\n")
        f.write(f"Jarque_Bera_pvalor: {p_jb:.4f}\n")
        
        f.write("Test ADF:\n")
        f.write(f"p-valor: {p_adf:.4f}\n")

    """Genera los gráficos de ACF y PACF del residuo para evaluar la autocorrelación."""

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(residuo_limpio, ax=axes[0])
    axes[0].set_title("ACF del residuo")

    plot_pacf(residuo_limpio, ax=axes[1], method='ywm')
    axes[1].set_title("PACF del residuo")

    plt.tight_layout()
    fig.savefig("output/ej4_acf_pacf.png", dpi=150, bbox_inches='tight')

    """Genera un histograma del residuo y superpone la curva de una distribución normal ajustada"""
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(residuo_limpio, bins=30, density=True, alpha=0.6)

    x = np.linspace(residuo_limpio.min(), residuo_limpio.max(), 100)
    y = norm.pdf(x, media, std)
    ax.plot(x, y)

    ax.set_title("Histograma del residuo + Normal ajustada")

    fig.savefig("output/ej4_histograma_ruido.png", dpi=150, bbox_inches='tight')


# =============================================================================
# MAIN — Ejecuta el pipeline completo
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("EJERCICIO 4 — Análisis de Series Temporales")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Paso 1: Generar la serie (NO modificar la semilla)
    # ------------------------------------------------------------------
    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print(f"\nSerie generada:")
    print(f"  Periodo:      {serie.index[0].date()} → {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media:         {serie.mean():.2f}")
    print(f"  Std:           {serie.std():.2f}")
    print(f"  Min / Max:     {serie.min():.2f} / {serie.max():.2f}")

    # ------------------------------------------------------------------
    # Paso 2: Visualizar la serie completa
    # ------------------------------------------------------------------
    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    # ------------------------------------------------------------------
    # Paso 3: Descomponer
    # ------------------------------------------------------------------
    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    # ------------------------------------------------------------------
    # Paso 4: Analizar el residuo
    # ------------------------------------------------------------------
    print("[3/3] Analizando el residuo...")
    if resultado is not None:
        analizar_residuo(resultado.resid)

    # ------------------------------------------------------------------
    # Resumen de salidas esperadas
    # ------------------------------------------------------------------
    print("\nSalidas esperadas en output/:")
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "✓" if existe else "✗ (pendiente)"
        print(f"  [{estado}] output/{s}")

    print("\n¡Recuerda completar las respuestas en Respuestas.md!")
