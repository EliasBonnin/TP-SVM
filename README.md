# Clasificación de Pokémon Legendarios con SVM

Este proyecto implementa un modelo de Machine Learning supervisado basado en Support Vector Machines (SVM) para predecir si un Pokémon es legendario (`is_legendary`) a partir de sus estadísticas de combate numéricas.

## Procedimiento General del Proyecto

### Fase 1: Carga y Preparación de Datos
1. **Carga del dataset:** Lectura del archivo `pokemon_complete_stats.csv`.
2. **Selección de variables:** Para evitar un modelo innecesariamente complejo y permitir una correcta interpretación visual mediante gráficos de 2 dimensiones, se realizó una reducción de características para utilizar únicamente las 2 variables de mayor impacto predictivo:
   * **Predictoras ($X$):** `base_experience` (Experiencia Base) y `special_attack` (Ataque Especial).
   * **Objetivo ($y$):** `is_legendary` (0 = No Legendario, 1 = Legendario).
3. **Limpieza de valores nulos:** Se descartaron los registros con datos incompletos en las variables de interés, garantizando la calidad del entrenamiento.

### Fase 2: Particionado y Escalado
4. **División Train / Test (80% / 20%):** Se realizó la separación de datos aplicando estratificación (`stratify=y`) para mantener la proporción de la clase minoritaria (legendarios, ~9%) equitativa en ambos conjuntos.
5. **Estandarización (`StandardScaler`):** Se ajustaron los datos (`fit_transform`) exclusivamente sobre el conjunto de entrenamiento y luego se transformó el conjunto de prueba para prevenir la fuga de información (*data leakage*).

### Fase 3: Experimentación y Comparativa de Kernels
Se entrenaron y evaluaron 3 configuraciones de kernel. En todas se aplicó el parámetro `class_weight='balanced'` para contrarrestar de forma automática el desbalance natural de las clases:

| Kernel | Geometría de la Frontera | Falsos Positivos (FP) | Falsos Negativos (FN) | Recall (Legendarios) | F1-Score | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RBF (Base)** | Campanas gaussianas locales | 29 | 1 | 95.8% | 0.605 | 88.5% |
| **Lineal** | Hiperplano (línea recta en 2D) | 42 | **0** | **100.0%** | 0.533 | 83.9% |
| **Polinomial (Grado 3)** | Curvas suaves y polinómicas | **21** | 2 | 91.7% | **0.657** | **91.2%** |

## Métricas Clave y Conclusiones 

* **El problema de la Precisión Global (Accuracy):** Al haber solo un ~9% de Pokémon legendarios en el dataset, guiarse solo por el Accuracy es engañoso. Un modelo que prediga que *ningún* Pokémon es legendario obtendría un 91% de exactitud a pesar de ser inútil.
* **Recall (Sensibilidad):** El **Kernel Lineal** obtiene un desempeño perfecto detectando legendarios (Recall 100%, 0 falsos negativos), asegurando que no se pierda ninguno, aunque a un gran costo: clasifica a 42 Pokémon comunes como legendarios (falsos positivos).
* **F1-Score (Equilibrio):** El **Kernel Polinomial de grado 3** resulta ser el modelo más robusto y equilibrado para este problema. Con el mejor puntaje F1 (0.657), logra mantener extremadamente bajos tanto los falsos negativos (2) como los falsos positivos (21), obteniendo simultáneamente la mejor exactitud global (91.2%).

## Visualización Gráfica

El proyecto genera dos visualizaciones principales para facilitar la interpretación del rendimiento espacial y métrico del modelo:
1. **Fronteras de Decisión en 2D (`fronteras_decision.png`):** Utiliza la técnica de *meshgrid* para colorear las áreas del plano en las que el modelo predice cada clase. Permite observar claramente la forma de la región de decisión establecida por cada kernel sobre las variables `base_experience` y `special_attack`, junto con el posicionamiento real de los datos del conjunto de pruebas.
2. **Comparativa de Kernels (`evaluacion_kernels_comparativa.png`):** Un panel que reúne las matrices de confusión generadas por las predicciones de cada kernel junto con una gráfica de curvas ROC superpuestas para contrastar el área bajo la curva (AUC).

## Ejecución del Proyecto

```bash
python3 practicaSVM.py
```
Durante la ejecución, la terminal mostrará la salida de la limpieza de datos, las dimensiones de la división, reportes de clasificación detallados (precision, recall, f1-score) y matrices de confusión en modo texto para cada kernel evaluado. Finalmente, se guardarán y desplegarán automáticamente en pantalla las dos imágenes comparativas mencionadas en el apartado gráfico.
