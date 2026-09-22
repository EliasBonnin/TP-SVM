# Clasificación de Pokémon Legendarios con SVM

Este proyecto implementa un modelo de Machine Learning supervisado basado en **Support Vector Machines (SVM)** para predecir si un Pokémon es legendario (`is_legendary`) a partir de sus estadísticas físicas y de combate numéricas.

---

## Procedimiento General del Proyecto

### Fase 1: Carga y Preparación de Datos
1. **Carga del dataset:** Lectura de `pokemon_complete_stats.csv`.
2. **Selección de variables:**
   * **Predictoras ($X$):** 9 características numéricas (`hp`, `attack`, `defense`, `special_attack`, `special_defense`, `speed`, `height_dm`, `weight_hg`, `base_experience`).
   * **Objetivo ($y$):** `is_legendary` (0 = Común, 1 = Legendario).
3. **Limpieza de valores nulos:** Se descartan los 49 Pokémon incompletos en `base_experience` para garantizar datos 100% limpios y sincronizados (1302 registros finales).

### Fase 2: Particionado y Escalado
4. **División Train / Test (80% / 20%):** Con **estratificación** (`stratify=y`) para mantener la proporción de legendarios (~9%) en ambos conjuntos.
5. **Estandarización (`StandardScaler`):** Ajuste (`fit_transform`) exclusivamente en Train y transformación (`transform`) en Test para prevenir fuga de datos (*data leakage*).

### Fase 3: Experimentación y Comparativa de Kernels
Se evalúan 3 configuraciones de kernel, todas con `class_weight='balanced'` debido al desbalance de clases:

| Kernel | Geometría de la Frontera | Falsos Positivos (FP) | Falsos Negativos (FN) | Recall (Legendarios) | F1-Score | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RBF (Base)** | Campanas gaussianas locales (dimensión infinita) | 24 | 5 | 79.2% | 0.567 | 88.9% |
| **Lineal** | Hiperplano plano en espacio original | 37 | **1** | **95.8%** | 0.548 | 85.4% |
| **Polinomial (Grado 3)** | Curvas por interacciones de variables | **24** | **3** | **87.5%** | **0.609** | **89.7%** |

---

## Conclusiones para la Defensa en Clase

* **Lineal:** Es el modelo con mayor **Recall (95.8%)** (casi no se le escapan legendarios, solo 1 FN), pero tiene el costo de acumular muchos **Falsos Positivos (37)**.
* **Polinomial (Grado 3):** Es el modelo con **mejor desempeño global**, logrando el **F1-Score más alto (0.609)** y un **Accuracy de 89.7%**, reduciendo los falsos negativos a solo 3 sin aumentar los falsos positivos.

---

## Ejecución

```bash
python3 practicaSVM.py
```
El script imprimirá en consola las matrices de confusión individuales, los reportes de clasificación y la **tabla resumen comparativa**. Además, generará y guardará la imagen con las 3 matrices visuales y la comparación de curvas ROC en `evaluacion_kernels_comparativa.png`.
