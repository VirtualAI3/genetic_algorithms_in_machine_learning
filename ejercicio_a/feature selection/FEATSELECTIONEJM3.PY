# ===============================
# IMPORTS
# ===============================
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import os

# ===============================
# 1. CARGA DEL DATASET
# ===============================
data = pd.read_csv(r"archive/santander.csv")  # raw string evita warning

X = data.drop(["target", "ID_code"], axis=1)
y = data["target"]

print("Shape del dataset:", X.shape)

# Escalado para LR y MLP
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ===============================
# 2. BASELINE CON RANDOM FOREST
# ===============================
print("\n===== BASELINE: RANDOM FOREST =====")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
baseline_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    score = rf.score(X.iloc[test_idx], y.iloc[test_idx])
    baseline_scores.append(score)
    print(f"Fold {fold} -> Accuracy: {score:.4f}")

baseline_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
print(f"\nROC AUC (Random Forest con todas las features): {baseline_auc:.4f}")

# ===============================
# 3. ALGORITMO GENÉTICO PARA FEATURE SELECTION
# ===============================
n_features = X.shape[1]
population_size = 5
generations = 5 
mutation_rate = 0.1
alpha = 0.05  # penalización por número de features

# Función de evaluación
def evaluate(individual, model="RF", show=False):
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected) == 0:
        return 0, 0
    
    X_sel = X.iloc[:, selected] if model not in ["MLP", "LR"] else X_scaled.iloc[:, selected]
    
    if model == "RF":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model == "XGB":
        clf = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    elif model == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=2000, random_state=42)
    elif model == "LGBM":
        clf = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model == "LR":
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
    else:
        raise ValueError("Modelo no soportado")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc = cross_val_score(clf, X_sel, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    penalty = alpha * (len(selected) / n_features)
    fitness = auc - penalty
    if show:
        print(f"   -> AUC sin penalizar: {auc:.4f} | Penalización: {penalty:.4f} | Fitness: {fitness:.4f}")
    return auc, fitness

# Inicialización
def init_population():
    return [np.random.randint(0, 2, n_features).tolist() for _ in range(population_size)]

def tournament_selection(pop, scores, k=3):
    selected = random.sample(range(len(pop)), k)
    best = max(selected, key=lambda idx: scores[idx])
    return pop[best]

def crossover(p1, p2):
    point = random.randint(1, n_features - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def mutate(ind):
    for i in range(n_features):
        if random.random() < mutation_rate:
            ind[i] = 1 - ind[i]
    return ind

# ===============================
# 4. EJECUCIÓN DEL ALGORITMO GENÉTICO PARA TODOS LOS MODELOS
# ===============================
models = ["RF", "XGB", "MLP", "LGBM", "LR"]
results = []

for model in models:
    print(f"\n===== ALGORITMO GENÉTICO CON {model} =====")
    population = init_population()
    best_solution = None
    best_score = -1
    history = []
    best_auc_unpenalized = 0

    for gen in range(generations):
        print(f"\n--- Generación {gen+1} ---")
        scores = []
        aucs = []
        for idx, ind in enumerate(population):
            print(f"Individuo {idx+1}: {sum(ind)} features seleccionadas")
            auc, fitness = evaluate(ind, model=model, show=True)
            scores.append(fitness)
            aucs.append(auc)
        
        gen_best_idx = scores.index(max(scores))
        gen_best = scores[gen_best_idx]
        gen_best_ind = population[gen_best_idx]
        gen_best_auc = aucs[gen_best_idx]
        
        if gen_best > best_score:
            best_score = gen_best
            best_solution = gen_best_ind
            best_auc_unpenalized = gen_best_auc
        
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
    
    selected_features = [i for i, bit in enumerate(best_solution) if bit == 1]
    avg_fitness = np.mean(history)
    
    results.append({
        "Modelo": model,
        "Fitness_final": best_score,
        "AUC_sin_penalizar": best_auc_unpenalized,
        "Features_seleccionadas": len(selected_features),
        "Promedio_fitness": avg_fitness
    })
    
    # Guardar CSV por modelo
    df_history = pd.DataFrame({
        "Generación": list(range(1, generations+1)),
        "Fitness": history
    })
    os.makedirs("resultados", exist_ok=True)
    df_history.to_csv(f"resultados/fitness_{model}.csv", index=False)
    
    # Gráfica
    plt.figure(figsize=(8,4))
    plt.plot(range(1, generations+1), history, marker="o")
    plt.title(f"Evolución del Fitness ({model})")
    plt.xlabel("Generación")
    plt.ylabel("Fitness (AUC penalizado)")
    plt.grid(True)
    plt.show()

# ===============================
# 5. TABLA COMPARATIVA FINAL
# ===============================
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Fitness_final", ascending=False)
print("\n==============================")
print("TABLA COMPARATIVA FINAL ENTRE MODELOS")
print(df_results)

# Guardar CSV resumen
df_results.to_csv("resultados/comparativa_final_modelos.csv", index=False)
