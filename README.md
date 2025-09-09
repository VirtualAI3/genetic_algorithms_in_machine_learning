# 🧬 Genetic Algorithms in Machine Learning

Este proyecto implementa diferentes aplicaciones de **Algoritmos Genéticos (GA)** en problemas de **aprendizaje automático**.  
El objetivo es aprovechar el poder de la optimización evolutiva para encontrar mejores configuraciones en modelos de machine learning.

---

## 📌 Módulos principales

### 🔹 FeatureSelection
Implementa un algoritmo genético para seleccionar el subconjunto óptimo de características, buscando un balance entre rendimiento del modelo y reducción de dimensionalidad.

Este módulo implementa un algoritmo Genético (GA) para la selección de características en datasets de alta dimensionalidad.

El objetivo es encontrar un subconjunto óptimo de features que maximice el rendimiento de los modelos mientras se minimiza la complejidad (número de variables).
✅ Ejemplo de uso:
```python
from feature_selection import GeneticFeatureSelector
import pandas as pd

# Cargar dataset Santander
data = pd.read_csv("archive/santander.csv")
X = data.drop(["target", "ID_code"], axis=1)
y = data["target"]

# Crear selector genético
selector = GeneticFeatureSelector(
    models=["rf", "xgb", "mlp", "svm", "lr"],
    population_size=20,
    generations=10,
    mutation_rate=0.1,
    alpha=0.05
)

# Ejecutar el proceso
results = selector.run(X, y)

# Exportar resultados
selector.save_results("results_feature_selection.csv")

# Visualización comparativa
selector.plot_results()
```



### 🔹 HyperparameterOptimizer
Un optimizador de hiperparámetros basado en algoritmos genéticos que permite:
- Explorar espacios de búsqueda complejos de parámetros.
- Soportar parámetros de tipo **real, discreto y categórico**.
- Evaluar configuraciones mediante **validación cruzada**.
- Utilizar **múltiples métricas de fitness** (ej: `accuracy`, `f1`, `roc_auc`) y combinarlas con diferentes pesos.
- Reportar la evolución de los mejores individuos por generación.
- Entrenar y evaluar automáticamente el mejor modelo encontrado.

✅ Ejemplo de uso:
```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score

param_ranges = {
    'hidden_layer_sizes': ([(50,), (100,), (50,50)], None, 'choice'),
    'activation': (['relu', 'tanh'], None, 'choice'),
    'alpha': (1e-5, 1e-1, 'float')
}

scoring = {
    "f1": (f1_score, 0.6),
    "accuracy": (accuracy_score, 0.4)
}

optimizer = HyperparameterOptimizer(
    model_class=MLPClassifier,
    X=X_train, y=y_train,
    param_ranges=param_ranges,
    scoring=scoring,
    cv_folds=5,
    generations=20
)

best_params = optimizer.optimize()
results = optimizer.evaluate_best_model(X_test, y_test)
````
### 🔹 NeuronEvolution
