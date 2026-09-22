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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
# PASO 5: División en Entrenamiento (Train) y Prueba (Test)
# ==============================================================================
# stratify=y preserva la proporción del target (~9% legendarios)
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
# fit_transform() SOLO en Train, transform() en Test (evita data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("⚖️  Características escaladas correctamente (media ~0, varianza ~1).")

# ==============================================================================
# PASO 7: FASE 3 - Entrenamiento y evaluación de diferentes kernels
# ==============================================================================
modelos = {
    "RBF (Base)": SVC(kernel="rbf", class_weight="balanced", random_state=42),
    "Lineal": SVC(kernel="linear", class_weight="balanced", random_state=42),
    "Polinomial (Grado 3)": SVC(kernel="poly", degree=3, class_weight="balanced", random_state=42),
}

target_names = ["No Legendario (0)", "Legendario (1)"]
predicciones = {}
tabla_metricas = []

for nombre, modelo in modelos.items():
    print("\n" + "=" * 60)
    print(f"🔬 ENTRENANDO Y EVALUANDO KERNEL: {nombre}")
    print("=" * 60)

    # Entrenamiento
    modelo.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = modelo.predict(X_test_scaled)
    predicciones[nombre] = y_pred

    # Matriz de confusión y reporte en texto
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Matriz de Confusión ({nombre}):")
    print(cm)
    print(f"\nReporte de Clasificación ({nombre}):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Almacenar métricas para la tabla comparativa
    tabla_metricas.append({
        "Kernel": nombre,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1-Score": round(f1_score(y_test, y_pred), 3),
        "Falsos Positivos (FP)": fp,
        "Falsos Negativos (FN)": fn,
    })

# ==============================================================================
# PASO 8: Tabla resumen comparativa
# ==============================================================================
df_comparativa = pd.DataFrame(tabla_metricas)
print("\n" + "=" * 70)
print("📊 RESUMEN COMPARATIVO DE KERNELS (FASE 3)")
print("=" * 70)
print(df_comparativa.to_string(index=False))
print("=" * 70)

# ==============================================================================
# PASO 9: Visualización gráfica comparativa
# ==============================================================================
# 3 matrices de confusión + 1 gráfico con curvas ROC superpuestas
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

for idx, (nombre, modelo) in enumerate(modelos.items()):
    y_pred = predicciones[nombre]
    cm = confusion_matrix(y_test, y_pred)

    # Matriz de Confusión visual para cada kernel
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Leg.", "Leg."])
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(f"MC: {nombre}")

    # Curvas ROC superpuestas en el 4to subplot
    RocCurveDisplay.from_estimator(
        modelo,
        X_test_scaled,
        y_test,
        name=nombre,
        ax=axes[3]
    )

axes[3].plot([0, 1], [0, 1], "k--", label="Aleatorio (AUC = 0.50)")
axes[3].set_title("Comparación de Curvas ROC")
axes[3].grid(True, linestyle="--", alpha=0.5)
axes[3].legend()

plt.tight_layout()
archivo_grafico = "evaluacion_kernels_comparativa.png"
plt.savefig(archivo_grafico, dpi=200)
print(f"\n📈 Gráficos comparativos guardados como '{archivo_grafico}'.")
plt.show()
