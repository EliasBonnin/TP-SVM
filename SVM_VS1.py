# Código actualizado para practicaSVM.py

import pandas as pd

# ------------------------------------------------------------------------------
# 1. Cargar el dataset
# ------------------------------------------------------------------------------
nombre_archivo = "pokemon_complete_stats.csv"
df = pd.read_csv(nombre_archivo)

filas_iniciales = len(df)
print(f"Dimensiones iniciales del dataset: {filas_iniciales} filas.")

# ------------------------------------------------------------------------------
# 2. Definir columnas de interés
# ------------------------------------------------------------------------------
features = [
    "base_experience",
    "special_attack",
]
target = "is_legendary"

# ------------------------------------------------------------------------------
# 3. Descartar filas con valores nulos (asegurando sincronía entre X e y)
# ------------------------------------------------------------------------------
# Eliminamos cualquier fila que tenga nulos en nuestras variables de interés
df_limpio = df.dropna(subset=features + [target]).copy()

filas_finales = len(df_limpio)
descartados = filas_iniciales - filas_finales

print(f"🧹 Filas descartadas con nulos: {descartados}")
print(f"✅ Filas finales disponibles: {filas_finales}")

# ------------------------------------------------------------------------------
# 4. Crear X e y a partir del DataFrame limpio
# ------------------------------------------------------------------------------
X = df_limpio[features].copy()
y = df_limpio[target].astype(int)

# Verificación de control:
print(f"\n¿Quedan nulos en X?: {X.isnull().sum().sum()}")
print(
    f"Distribución del target (0 = Normal, 1 = Legendario):\n{y.value_counts()}"
)
