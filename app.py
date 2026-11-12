import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Regresión lineal múltiple",
    page_icon=":material/water_drop:",
    layout="wide",
)
st.title("Regresión lineal múltiple")
st.caption("Análisis del consumo de agua con datos simulados en DuckDB.")

# ==============================================================================
# PASO 1: SIMULACIÓN DE DATOS RELACIONALES EN DUCKDB (DETERMINISTA)
# ==============================================================================
con = duckdb.connect(database=':memory:')

# Fijar semilla determinista en DuckDB
con.execute("SELECT setseed(0.42);")

# Crear tabla de datos simulados
con.execute("""
CREATE TABLE datos_agua AS
SELECT 
    row_number() OVER () AS residencia_id,
    ROUND(18 + RANDOM() * 17, 2) AS temperatura_max,     -- 18°C a 35°C
    ROUND(40 + RANDOM() * 45, 2) AS humedad_max,         -- 40% a 85%
    ROUND(2 + RANDOM() * 5, 1) AS personas_familia,       -- 2 a 7 personas
    ROUND(0.3 + RANDOM() * 1.5, 2) AS consumo_bidon_pc,   -- 0.3 a 1.8 lt/hab-día
    ROUND(
        94.49 
        + (2.48 * (18 + RANDOM() * 17)) 
        - (0.79 * (40 + RANDOM() * 45)) 
        - (5.18 * (2 + RANDOM() * 5)) 
        + (14.89 * (0.3 + RANDOM() * 1.5)) 
        + (RANDOM() * 4 - 2), 2                           -- Ruido aleatorio
    ) AS cpc_consumo
FROM generate_series(1, 100);
""")

# Extraer el dataset a DataFrame de Pandas
df_agua = con.execute("SELECT * FROM datos_agua").df()

st.header("1. Datos simulados")
st.dataframe(
    df_agua.head(),
    hide_index=True,
    alt="Primeras cinco filas de los datos simulados de consumo de agua",
)

# ==============================================================================
# PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r)
# ==============================================================================
columnas_interes = ['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc', 'cpc_consumo']
matriz_correlacion = df_agua[columnas_interes].corr()
correlacion_respuesta = matriz_correlacion['cpc_consumo'].drop('cpc_consumo')

st.header("2. Matriz de correlación de Pearson")
st.dataframe(
    matriz_correlacion,
    alt="Correlación de Pearson entre las variables del modelo",
)
st.subheader("Correlación con el consumo")
st.dataframe(
    correlacion_respuesta.rename("Correlación").to_frame(),
    alt="Correlación de cada variable explicativa con el consumo",
)

# ==============================================================================
# PASO 3: AJUSTE DEL MODELO OLS (statsmodels)
# ==============================================================================
X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
y = df_agua['cpc_consumo']

# Agregar término constante (intersección)
X_const = sm.add_constant(X)

# Ajustar el modelo OLS
modelo = sm.OLS(y, X_const).fit()

st.header("3. Modelo de regresión OLS")
st.code(modelo.summary().as_text(), language=None)

# ==============================================================================
# PASO 4: DIAGNÓSTICO DE RESIDUOS Y SUPUESTOS
# ==============================================================================
residuos = modelo.resid
y_pred = modelo.fittedvalues

# 1. Prueba de Normalidad (Shapiro-Wilk)
stat_shapiro, p_val_shapiro = stats.shapiro(residuos)
st.header("4. Diagnóstico de residuos")
shapiro_col, p_value_col, dw_col = st.columns(3)
shapiro_col.metric("Estadístico Shapiro-Wilk", f"{stat_shapiro:.4f}")
p_value_col.metric("Valor p", f"{p_val_shapiro:.4f}")

if p_val_shapiro > 0.05:
    st.success("Los residuos cumplen con el supuesto de normalidad (p > 0.05).")
else:
    st.warning("Los residuos no cumplen con el supuesto de normalidad (p <= 0.05).")

# 2. Estadístico Durbin-Watson (Independencia)
dw = sm.stats.stattools.durbin_watson(residuos)
dw_col.metric("Estadístico Durbin-Watson", f"{dw:.4f}")
st.caption("Valores entre 1.5 y 2.5 indican ausencia de autocorrelación.")

# ==============================================================================
# PASO 5: MÉTRICAS DE RENDIMIENTO DEL MODELO (scikit-learn)
# ==============================================================================
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

st.header("5. Rendimiento del modelo")
r2_col, rmse_col = st.columns(2)
r2_col.metric("Coeficiente de determinación (R²)", f"{r2 * 100:.2f}%")
rmse_col.metric("Error cuadrático medio raíz (RMSE)", f"{rmse:.4f} lt/hab-día")
params = modelo.params
ecuacion_modelo = (
    f"CPC = {params['const']:.2f} "
    f"+ ({params['temperatura_max']:.2f} * Temp) "
    f"+ ({params['humedad_max']:.2f} * Humedad) "
    f"+ ({params['personas_familia']:.2f} * Personas) "
    f"+ ({params['consumo_bidon_pc']:.2f} * Bidón)"
)
st.subheader("Ecuación del modelo ajustado")
st.code(ecuacion_modelo, language=None)

# ==============================================================================
# PASO 6: GENERACIÓN DEL PANEL DE DIAGNÓSTICO VISUAL (5 GRÁFICAS)
# ==============================================================================
st.header("6. Panel de diagnóstico visual")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)', fontsize=16, fontweight='bold')

# 1. Mapa de Calor de Correlación
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

# 2. Scatter Plot: Humedad vs Consumo
sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1],
            line_kws={'color': 'red', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Humedad Máx vs Consumo')
axes[0, 1].set_xlabel('Humedad Máxima (%)')
axes[0, 1].set_ylabel('CPC Consumo (lt/hab-día)')

# 3. Valores Reales vs Predichos
axes[0, 2].scatter(y_pred, y, alpha=0.7, color='teal')
axes[0, 2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2*100:.1f}%)')
axes[0, 2].set_xlabel('CPC Predicho (ŷ)')
axes[0, 2].set_ylabel('CPC Real (y)')

# 4. Residuos vs Valores Ajustados (Homocedasticidad)
axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')
axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

# 5. Gráfico Q-Q Normal (Normalidad de Errores)
sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

# Ocultar la celda sobrante (posición [1, 2])
axes[1, 2].axis('off')

plt.tight_layout()
st.pyplot(
    fig,
    alt="Panel con cinco gráficos de diagnóstico de la regresión lineal",
)