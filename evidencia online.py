import pandas as pd
import numpy as np
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# Función para descargar y leer CSV desde Google Drive usando gdown
def URLBd(file_id, output_name):
    if not os.path.exists(output_name):  # Solo descarga si no existe
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)
    return pd.read_csv(output_name)

# Descargar y cargar los archivos
insights = URLBd("1WHl94mNSdHawCbf7a9KF7MjWeZsqUTYc", "insights.csv")
predicciones_2024 = URLBd("1v_SHwcpj0alS82Xdwkqp-CC5FDCn2_QL", "predicciones_2024.csv")

# Preprocesamiento
insights['FECHA_HORA_COMPLETA_DEL_HECHO'] = pd.to_datetime(insights['FECHA_HORA_COMPLETA_DEL_HECHO'])
datos_2023 = insights[insights['FECHA_HORA_COMPLETA_DEL_HECHO'].dt.year == 2023].copy()

print("\n--- Estadística Descriptiva ---")
print(datos_2023[['LATITUD', 'LONGITUD']].describe())

label_encoder = LabelEncoder()
datos_2023['DELITO_LABEL'] = label_encoder.fit_transform(datos_2023['CATEGORIA_DEL_DELITO'])

X = datos_2023[['LATITUD', 'LONGITUD']]
y = datos_2023['DELITO_LABEL']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

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

# Análisis con statsmodels
df_val = X_val.copy()
df_val['y'] = y_val.values

modelo_formula = smf.ols('y ~ LATITUD + LONGITUD', data=df_val).fit()
anova = sm.stats.anova_lm(modelo_formula, typ=2)

print("\n--- ANOVA ---")
print(anova)

print("\n--- Intervalos de Confianza ---")
print(modelo_formula.conf_int())


