# Clasificación de Pokémon Legendarios con SVM

Este proyecto implementa un modelo de Machine Learning supervisado basado en **Support Vector Machines (SVM)** para predecir si un Pokémon es legendario (`is_legendary`) en función de sus atributos físicos y estadísticas de combate.

---

## Procedimiento General

1. **Carga y Selección de Variables**
   * Se lee el dataset `pokemon_complete_stats.csv`.
   * **Variables predictoras ($X$):** 9 características numéricas (`hp`, `attack`, `defense`, `special_attack`, `special_defense`, `speed`, `height_dm`, `weight_hg`, `base_experience`).
   * **Variable objetivo ($y$):** `is_legendary` convertida a valores binarios (0 = No Legendario, 1 = Legendario).

2. **Limpieza de Datos**
   * Se identifican y descartan los 49 registros que contenían valores nulos en `base_experience`, garantizando que $X$ e $y$ queden perfectamente alineados y sin datos incompletos (1302 Pokémon finales).

3. **División de Datos (Train / Test)**
   * Se divide el dataset en 80% para entrenamiento (1041 muestras) y 20% para evaluación (261 muestras).
   * Se aplica **estratificación** (`stratify=y`) para preservar el porcentaje real de legendarios (~9%) en ambos subconjuntos frente al desbalance de clases.

4. **Escalado de Características (`StandardScaler`)**
   * Las SVM son sensibles a las diferencias de magnitud entre variables (p. ej. peso vs velocidad).
   * Se estandarizan las características a media 0 y varianza 1.
   * Se aplica `fit_transform` solo en entrenamiento y `transform` en prueba para evitar fuga de información (*data leakage*).

5. **Entrenamiento del Modelo SVM**
   * Se utiliza un clasificador `SVC` con **kernel RBF** (Gaussiano) para capturar relaciones no lineales.
   * Se activa `class_weight='balanced'` para dar mayor peso a la clase minoritaria (legendarios) durante el cálculo del margen.

6. **Evaluación y Visualización Gráfica**
   * **Métricas:** Matriz de Confusión, Precisión, Recall y F1-Score.
   * **Gráficos generados (`evaluacion_svm.png`):**
     * **Matriz de Confusión Visual:** Muestra aciertos y errores de clasificación.
     * **Curva ROC y AUC:** Mide la capacidad global del modelo para distinguir entre ambas clases.

---

## Cómo ejecutar el script

```bash
python3 practicaSVM.py
```
El script mostrará las métricas en la terminal, abrirá la ventana con los gráficos y los guardará automáticamente como `evaluacion_svm.png`.
