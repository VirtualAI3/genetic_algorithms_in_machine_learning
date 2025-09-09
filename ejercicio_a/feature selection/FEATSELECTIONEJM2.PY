import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ==============================
# 1. CARGA DEL DATASET
# ==============================
# ⚠️ Arreglo del warning con raw string
data = pd.read_csv(r"archive/santander.csv")

# Variables y target
X = data.drop(["target", "ID_code"], axis=1)
y = data["target"]

print("Shape del dataset:", X.shape)

# ==============================
# 2. BASELINE CON RANDOM FOREST
# ==============================
print("\n===== BASELINE: RANDOM FOREST =====")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold_scores = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    score = rf.score(X.iloc[test_idx], y.iloc[test_idx])
    fold_scores.append(score)
    print(f"Fold {fold} -> Accuracy: {score:.4f}")

baseline_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
print(f"\nROC AUC (Random Forest con todas las features): {baseline_auc:.4f}")

# ==============================
# 3. ALGORITMO GENÉTICO PARA FEATURE SELECTION
# ==============================
n_features = X.shape[1]
population_size = 10
generations = 5
mutation_rate = 0.1
alpha = 0.05

# --- Función de evaluación ---
def evaluate(individual, show=False):
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected) == 0:
        return 0
    X_sel = X.iloc[:, selected]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc = cross_val_score(rf, X_sel, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    penalty = alpha * (len(selected) / n_features)
    fitness = auc - penalty
    if show:
        print(f"   -> AUC sin penalizar: {auc:.4f} | Penalización: {penalty:.4f}")
    return fitness

# --- Inicialización ---
def init_population():
    return [np.random.randint(0, 2, n_features).tolist() for _ in range(population_size)]

# --- Selección por torneo ---
def tournament_selection(pop, scores, k=3):
    selected = random.sample(range(len(pop)), k)
    best = max(selected, key=lambda idx: scores[idx])
    return pop[best]

# --- Crossover ---
def crossover(p1, p2):
    point = random.randint(1, n_features - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

# --- Mutación ---
def mutate(ind):
    for i in range(n_features):
        if random.random() < mutation_rate:
            ind[i] = 1 - ind[i]
    return ind

# --- Algoritmo Genético ---
population = init_population()
best_solution = None
best_score = -1
history = []

print("\n===== INICIO DEL ALGORITMO GENÉTICO =====")
for gen in range(generations):
    print(f"\n--- Generación {gen+1} ---")
    scores = []
    for idx, ind in enumerate(population):
        print(f"Individuo {idx+1}: {sum(ind)} features seleccionadas")
        fitness = evaluate(ind, show=True)
        scores.append(fitness)
        print(f"   -> Fitness (AUC penalizado) = {fitness:.4f}")
    
    gen_best = max(scores)
    gen_best_ind = population[scores.index(gen_best)]
    
    if gen_best > best_score:
        best_score = gen_best
        best_solution = gen_best_ind
    
    history.append(gen_best)
    print(f"✔ Mejor de la generación {gen+1}: Fitness={gen_best:.4f} | Features={sum(gen_best_ind)}")
    
    # Nueva población
    new_pop = []
    while len(new_pop) < population_size:
        p1 = tournament_selection(population, scores)
        p2 = tournament_selection(population, scores)
        c1, c2 = crossover(p1, p2)
        new_pop.append(mutate(c1))
        new_pop.append(mutate(c2))
    population = new_pop[:population_size]

# ==============================
# 4. RESULTADOS
# ==============================
selected_features = [i for i, bit in enumerate(best_solution) if bit == 1]

print("\n==============================")
print("RESULTADOS FINALES")
print("==============================")
print(f"ROC AUC (baseline RF con todas las features): {baseline_auc:.4f}")
print("Mejor subconjunto encontrado:")
print("Features seleccionadas:", len(selected_features))
print("Índices:", selected_features)
print(f"Fitness final (AUC penalizado): {best_score:.4f}")

# ==============================
# 5. VISUALIZACIÓN
# ==============================
plt.figure(figsize=(8,4))
plt.plot(range(1, generations+1), history, marker="o")
plt.title("Evolución del Fitness (Algoritmo Genético)")
plt.xlabel("Generación")
plt.ylabel("Fitness (AUC penalizado)")
plt.grid(True)
plt.show()
