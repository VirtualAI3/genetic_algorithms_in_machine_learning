import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hyperparameter_optimizer import HyperparameterOptimizer

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def example_diabetes_dataset():
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Cargar dataset diabetes desde CSV
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    cols = ['preg', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'target']
    data = pd.read_csv(url, names=cols)
    
    # Justo después de leer el dataset:
    cols_with_zeros = ['glucose', 'bp', 'skin', 'insulin', 'bmi']
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, pd.NA)

    # Imputación simple (podrías usar también KNN o IterativeImputer)
    data.fillna(data.median(), inplace=True)

    X = data.drop(columns=['target']).values
    y = data['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_ranges = {
        'hidden_layer_sizes': ([(50,), (100,), (50,50)], None, 'choice'),
        'activation': (['identity', 'logistic', 'tanh', 'relu'], None, 'choice'),
        'solver': (['lbfgs', 'adam', 'sgd'], None, 'choice'),
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
        X=X_train,
        y=y_train,
        param_ranges=param_ranges,
        scoring=scoring,
        cv_folds=5,
        population_size=20,
        generations=5,
        mutation_rate=0.3
    )

    best_params = optimizer.optimize()
    results = optimizer.evaluate_best_model(X_test, y_test)

    return optimizer, results


def plot_fitness_evolution(optimizer, title="Evolución del Fitness"):
    """Grafica la evolución del fitness a lo largo de las generaciones"""
    fitness_history_generation, fitness_history_global, fitness_avg_history = optimizer.get_fitness_history()
    
    plt.plot(fitness_history_generation, label="Fitness por Generación", linestyle='--', color='blue')
    plt.plot(fitness_history_global, label="Mejor Fitness Global", color='green')
    plt.plot(fitness_avg_history, label="Fitness Promedio por Generación", linestyle=':', color='red')
    plt.legend()
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)

    optimizer_dia, results_dia = example_diabetes_dataset()
    plot_fitness_evolution(optimizer_dia, "Diabetes - Evolución del Fitness")
