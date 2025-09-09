import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Cargar dataset Boston Housing
cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
data = pd.read_csv("archive\housing.csv", delim_whitespace=True, header=None, names=cols)

X = data.drop("MEDV", axis=1).values
y = data["MEDV"].values
n_features = X.shape[1]

# Función de aptitud
def fitness(ind):
    if np.sum(ind) == 0:  # evitar cromosomas vacíos
        return 0
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    score = cross_val_score(model, X[:, ind==1], y, cv=3, scoring="r2").mean()
    return score

# Inicialización
def init_population(size=6):
    return np.random.randint(0, 2, (size, n_features))

# Selección
def selection(pop, scores):
    idx = np.argsort(scores)[-2:]  # top 2
    return pop[idx]

# Cruzamiento
def crossover(p1, p2):
    point = np.random.randint(1, n_features-1)
    child = np.concatenate((p1[:point], p2[point:]))
    print(f"   > Crossover en punto {point}: {p1} + {p2} -> {child}")
    return child

# Mutación
def mutate(ind, rate=0.2):
    for i in range(n_features):
        if np.random.rand() < rate:
            ind[i] = 1 - ind[i]
            print(f"   > Mutación en gen {i}: {ind}")
    return ind

# Algoritmo Genético
pop = init_population(6)
generaciones = 5

for gen in range(generaciones):
    print("\n==============================")
    print(f" Generación {gen}")
    print("==============================")

    # Evaluar población
    scores = np.array([fitness(ind) for ind in pop])
    for i, (ind, score) in enumerate(zip(pop, scores)):
        print(f" Individuo {i}: {ind} | Features: {np.sum(ind)} | R2={score:.4f}")

    # Mejor individuo
    best_idx = np.argmax(scores)
    print(f" >>> Mejor individuo: {pop[best_idx]} con R2={scores[best_idx]:.4f}")

    # Selección
    parents = selection(pop, scores)
    print(f" Padres seleccionados:\n {parents[0]}\n {parents[1]}")

    # Cruzamiento + Mutación
    child1 = mutate(crossover(parents[0], parents[1]))
    child2 = mutate(crossover(parents[1], parents[0]))

    # Nueva población
    pop = np.vstack([parents, child1, child2, init_population(4)])

print("\n==============================")
print("PROCESO FINALIZADO")
final_scores = np.array([fitness(ind) for ind in pop])
best_idx = np.argmax(final_scores)
print(f"\nMejor solución encontrada: {pop[best_idx]}")
print(f"Features seleccionadas: {np.sum(pop[best_idx])}")
print(f"Mejor R2 alcanzado: {final_scores[best_idx]:.4f}")
