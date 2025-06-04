# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Cargar archivos (asegúrate de que los archivos estén en la misma carpeta que este script)
insights = pd.read_csv("Insights (1).csv")
predicciones_2024 = pd.read_csv("predicciones_delitos_2024_limpio.csv")

# Convertir fechas
insights['FECHA_HORA_COMPLETA_DEL_HECHO'] = pd.to_datetime(insights['FECHA_HORA_COMPLETA_DEL_HECHO'])

# Filtrar solo año 2023 para validación
datos_2023 = insights[insights['FECHA_HORA_COMPLETA_DEL_HECHO'].dt.year == 2023].copy()

# --- 1. Estadística Descriptiva ---
print("\n--- Estadística Descriptiva ---")
print(datos_2023[['LATITUD', 'LONGITUD']].describe())

# --- 2. Preparar datos para modelar ---
label_encoder = LabelEncoder()
datos_2023['DELITO_LABEL'] = label_encoder.fit_transform(datos_2023['CATEGORIA_DEL_DELITO'])

X = datos_2023[['LATITUD', 'LONGITUD']]
y = datos_2023['DELITO_LABEL']

# División: entrenamiento y validación (solo se usa para evaluación)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. Modelización con 4 regresores ---
modelos = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

print("\n--- Resultados de Modelos ---")
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{nombre}:")
    print(f" - MSE = {mse:.4f}")
    print(f" - R2 ajustado = {r2:.4f}\n")

# --- 4. ANOVA ---
import statsmodels.formula.api as smf

df_val = X_val.copy()
df_val['y'] = y_val.values

modelo_formula = smf.ols('y ~ LATITUD + LONGITUD', data=df_val).fit()
anova = sm.stats.anova_lm(modelo_formula, typ=2)

print("\n--- ANOVA ---")
print(anova)

# Intervalos de confianza
print("\n--- Intervalos de Confianza ---")
print(modelo_formula.conf_int())