import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hyperparameter_optimizer import HyperparameterOptimizer

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE


# =======================================================
# Función de preprocesamiento compartido
# =======================================================
def preprocess_diabetes_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    cols = ['preg', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'target']
    data = pd.read_csv(url, names=cols)

    # Reemplazar ceros por NA en variables fisiológicas
    cols_with_zeros = ['glucose', 'bp', 'skin', 'insulin', 'bmi']
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)

    # Separar X/y
    X = data.drop(columns=['target'])
    y = data['target']

    # Imputación KNN
    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(X)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Balanceo con SMOTE (solo en train)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Escalado
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# =======================================================
# MLP
# =======================================================
def example_diabetes_mlp():
    X_train, X_test, y_train, y_test = preprocess_diabetes_dataset()

    param_ranges = {
        'hidden_layer_sizes': ([(50,), (100,), (50, 50), (100, 50), (100, 100)], None, 'choice'),
        'activation': (['identity', 'logistic', 'tanh', 'relu'], None, 'choice'),
        'solver': (['adam', 'sgd'], None, 'choice'),
        'alpha': (0.0001, 0.1, 'float'),
        'learning_rate_init': (0.0001, 0.1, 'float')
    }

    scoring = {
        "f1": (f1_score, 0.6),
        "accuracy": (accuracy_score, 0.3),
        "roc_auc": (roc_auc_score, 0.1)
    }

    optimizer = HyperparameterOptimizer(
        model_class=MLPClassifier,
        fixed_params={"max_iter": 500, "early_stopping": True, "n_iter_no_change": 10, "random_state": 42},
        X=X_train,
        y=y_train,
        param_ranges=param_ranges,
        scoring=scoring,
        cv_folds=5,
        population_size=20,
        generations=10,
        mutation_rate=0.3
    )

    best_params = optimizer.optimize()
    results = optimizer.evaluate_best_model(X_test, y_test)

    return optimizer, results


# =======================================================
# XGBoost
# =======================================================
def example_diabetes_xgboost():
    X_train, X_test, y_train, y_test = preprocess_diabetes_dataset()

    param_ranges = {
        'n_estimators': (50, 300, 'int'),
        'max_depth': (3, 10, 'int'),
        'learning_rate': (0.01, 0.3, 'float'),
        'subsample': (0.5, 1.0, 'float'),
        'colsample_bytree': (0.5, 1.0, 'float'),
        'gamma': (0, 5, 'float'),
        'reg_alpha': (0, 1, 'float'),
        'reg_lambda': (0, 1, 'float')
    }

    scoring = {
        "f1": (f1_score, 0.6),
        "accuracy": (accuracy_score, 0.3),
        "roc_auc": (roc_auc_score, 0.1)
    }

    optimizer = HyperparameterOptimizer(
        model_class=XGBClassifier,
        X=X_train,
        y=y_train,
        param_ranges=param_ranges,
        scoring=scoring,
        cv_folds=5,
        population_size=20,
        generations=10,
        mutation_rate=0.3
    )

    best_params = optimizer.optimize()
    results = optimizer.evaluate_best_model(X_test, y_test)

    return optimizer, results


# =======================================================
# Random Forest
# =======================================================
def example_diabetes_randomforest():
    X_train, X_test, y_train, y_test = preprocess_diabetes_dataset()

    param_ranges = {
        'n_estimators': (50, 300, 'int'),
        'max_depth': (3, 20, 'int'),
        'min_samples_split': (2, 10, 'int'),
        'min_samples_leaf': (1, 5, 'int'),
        'max_features': (['sqrt', 'log2', None], None, 'choice')
    }

    scoring = {
        "f1": (f1_score, 0.6),
        "accuracy": (accuracy_score, 0.3),
        "roc_auc": (roc_auc_score, 0.1)
    }

    optimizer = HyperparameterOptimizer(
        model_class=RandomForestClassifier,
        X=X_train,
        y=y_train,
        param_ranges=param_ranges,
        scoring=scoring,
        cv_folds=5,
        population_size=20,
        generations=10,
        mutation_rate=0.3
    )

    best_params = optimizer.optimize()
    results = optimizer.evaluate_best_model(X_test, y_test)

    return optimizer, results


# =======================================================
# Plot fitness
# =======================================================
def plot_fitness_evolution(optimizer, ax, title="Evolución del Fitness"):
    fitness_generation, fitness_global, fitness_avg = optimizer.get_fitness_history()

    ax.plot(fitness_generation, label="Fitness por Generación", linestyle='--', color='blue')
    ax.plot(fitness_global, label="Mejor Fitness Global", color='green')
    ax.plot(fitness_avg, label="Fitness Promedio", linestyle=':', color='red')
    ax.set_title(title)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.grid(True)
    ax.legend()

# =======================================================
# Main
# =======================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS - MLPClassifier")
    print("=" * 60)
    optimizer_mlp, results_mlp = example_diabetes_mlp()

    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS - XGBoost")
    print("=" * 60)
    optimizer_xgb, results_xgb = example_diabetes_xgboost()

    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS - RandomForest")
    print("=" * 60)
    optimizer_rf, results_rf = example_diabetes_randomforest()
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plot_fitness_evolution(optimizer_mlp, ax1, "MLPClassifier")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    plot_fitness_evolution(optimizer_xgb, ax2, "XGBoost")

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    plot_fitness_evolution(optimizer_rf, ax3, "RandomForest")

    plt.show()  # Esto mostrará los 3 juntos
