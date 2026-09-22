import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# ==============================================================================
# PASO 1: Cargar el dataset
# ==============================================================================
nombre_archivo = "pokemon_complete_stats.csv"
df = pd.read_csv(nombre_archivo)
print(f"📊 Dataset cargado: {len(df)} registros totales.")

# ==============================================================================
# PASO 2: Definir variables predictoras (features) y variable objetivo (target)
# ==============================================================================
features = [
    "hp",
    "attack",
    "defense",
    "special_attack",
    "special_defense",
    "speed",
    "height_dm",
    "weight_hg",
    "base_experience",
]
target = "is_legendary"

# ==============================================================================
# PASO 3: Descartar filas con valores nulos (49 pokémon incompletos)
# ==============================================================================
df_limpio = df.dropna(subset=features + [target]).copy()
filas_descartadas = len(df) - len(df_limpio)
print(f"🧹 Filas descartadas con nulos: {filas_descartadas}")
print(f"✅ Filas limpias disponibles: {len(df_limpio)}")

# ==============================================================================
# PASO 4: Construir matrices X (características) e y (etiquetas 0 y 1)
# ==============================================================================
X = df_limpio[features].copy()
y = df_limpio[target].astype(int)

# ==============================================================================
# PASO 5: División en conjuntos de Entrenamiento (Train) y Prueba (Test)
# ==============================================================================
# Usamos stratify=y para mantener la misma proporción de legendarios en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n📂 División de datos:")
print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
print(f"   - Prueba (Test): {X_test.shape[0]} muestras")

# ==============================================================================
# PASO 6: Escalado de características (StandardScaler)
# ==============================================================================
# ¡Clave en ML!: fit_transform() SOLO en Train. En Test únicamente transform()
# para evitar fuga de información (data leakage).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("⚖️  Características escaladas correctamente (media ~0, varianza ~1).")

# ==============================================================================
# PASO 7: Definición y entrenamiento del modelo SVM
# ==============================================================================
# Usamos class_weight='balanced' para compensar que hay muchos menos legendarios
modelo_svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
modelo_svm.fit(X_train_scaled, y_train)

print("🤖 Modelo SVM entrenado con éxito.")

# ==============================================================================
# PASO 8: Evaluación numérica del modelo
# ==============================================================================
y_pred = modelo_svm.predict(X_test_scaled)

print("\n" + "=" * 55)
print("MATRIZ DE CONFUSIÓN")
print("=" * 55)
print(confusion_matrix(y_test, y_pred))

print("\n" + "=" * 55)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["No Legendario (0)", "Legendario (1)"]))

# ==============================================================================
# PASO 9: Visualización gráfica de resultados
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Gráfico de Matriz de Confusión
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Legendario", "Legendario"],
    cmap="Blues",
    ax=axes[0],
)
axes[0].set_title("Matriz de Confusión Visual")

# 2. Curva ROC (Evaluación de discriminación global)
RocCurveDisplay.from_estimator(
    modelo_svm,
    X_test_scaled,
    y_test,
    name="SVM (RBF)",
    ax=axes[1],
    color="darkorange",
)
axes[1].plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio (AUC = 0.50)")
axes[1].set_title("Curva ROC")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend()

plt.tight_layout()
archivo_grafico = "evaluacion_svm.png"
plt.savefig(archivo_grafico, dpi=200)
print(f"\n📈 Gráficos guardados exitosamente como '{archivo_grafico}'.")
plt.show()
