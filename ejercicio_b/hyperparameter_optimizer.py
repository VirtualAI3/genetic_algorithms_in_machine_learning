import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from genetic_algorithm import GeneticAlgorithm
from typing import Dict, Any

class HyperparameterOptimizer:
    def __init__(self, model_class, X, y, param_ranges: dict, 
                 scoring: Any = 'accuracy', cv_folds: int = 5,
                 population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """
        Optimizador de hiperparámetros usando algoritmos genéticos
        
        Args:
            model_class: Clase del modelo de sklearn (ej: RandomForestClassifier)
            X: Features del dataset
            y: Target del dataset
            param_ranges: Diccionario con rangos de parámetros
            scoring: Métrica de evaluación
            cv_folds: Número de folds para cross-validation
        """
        self.model_class = model_class
        self.X = X
        self.y = y
        self.param_ranges = param_ranges
        self.scoring = scoring
        self.cv_folds = cv_folds
        
        # Inicializar algoritmo genético
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
    
    def fitness_function(self, params: dict) -> float:
        """
        Función de fitness que evalúa un conjunto de parámetros.
        Si self.scoring es un string -> usa esa métrica.
        Si es un diccionario -> combina varias métricas con pesos.
        """
        try:
            model = self.model_class(**params)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

            # Caso 1: métrica única (string)
            if isinstance(self.scoring, str):
                cv_scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=self.scoring)
                return np.mean(cv_scores)

            # Caso 2: múltiples métricas con pesos
            elif isinstance(self.scoring, dict):
                total_score = 0.0
                for metric_name, (metric_func, weight) in self.scoring.items():
                    scorer = make_scorer(metric_func)
                    scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=scorer)
                    total_score += weight * np.mean(scores)
                return total_score

            else:
                raise ValueError("Formato de 'scoring' no soportado")

        except Exception as e:
            print(f"Error con parámetros {params}: {e}")
            return -1.0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Ejecuta la optimización de hiperparámetros
        
        Returns:
            Diccionario con los mejores parámetros encontrados
        """
        print("Iniciando optimización de hiperparámetros...")
        print(f"Espacio de búsqueda: {self.param_ranges}")
        print(f"Población: {self.ga.population_size}, Generaciones: {self.ga.generations}")
        print("-" * 50)
        
        # Ejecutar algoritmo genético
        best_params = self.ga.evolve(self.param_ranges, self.fitness_function)
        
        print(f"\nOptimización completada!")
        print(f"Mejor fitness alcanzado: {self.ga.best_fitness:.4f}")
        print(f"Mejores parámetros: {best_params}")
        
        return best_params

    def evaluate_best_model(self, X_test=None, y_test=None):
        """
        Evalúa el mejor modelo encontrado con las métricas especificadas.
        
        Si se proporcionan datos de test, evalúa todas las métricas usadas
        durante la optimización (accuracy, f1, roc_auc, etc.)
        """
        if self.ga.best_individual is None:
            print("Debe ejecutar optimize() primero")
            return None

        # Crear y entrenar el mejor modelo
        best_model = self.model_class(**self.ga.best_individual)
        best_model.fit(self.X, self.y)

        results = {
            'model': best_model,
            'params': self.ga.best_individual,
            'cv_score': self.ga.best_fitness
        }

        # Si se proporcionan datos de test, evaluar métricas
        if X_test is not None and y_test is not None:
            y_pred = best_model.predict(X_test)
            test_metrics = {}

            # Para ROC AUC, se necesita predict_proba o decision_function
            y_proba = None
            if hasattr(best_model, "predict_proba"):
                try:
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                except:
                    pass
            elif hasattr(best_model, "decision_function"):
                try:
                    y_proba = best_model.decision_function(X_test)
                except:
                    pass

            if isinstance(self.scoring, dict):
                for metric_name, (metric_func, _) in self.scoring.items():
                    try:
                        if metric_name == "roc_auc" and y_proba is not None:
                            score = metric_func(y_test, y_proba)
                        else:
                            score = metric_func(y_test, y_pred)
                        test_metrics[metric_name] = score
                        print(f"{metric_name} en test: {score:.4f}")
                    except Exception as e:
                        print(f"Error al calcular {metric_name} en test: {e}")
            else:
                # Solo accuracy si scoring no es dict
                score = accuracy_score(y_test, y_pred)
                test_metrics['accuracy'] = score
                print(f"Accuracy en test: {score:.4f}")

            results['test_metrics'] = test_metrics

        return results

    def get_fitness_history(self):
        """Retorna el historial de fitness para visualización"""
        return self.ga.fitness_history_generation, self.ga.fitness_history_global, self.ga.fitness_avg_history
    
    def print_generation_report(self, generation_num, top_n=5, show_all=False):
        """
        Imprime reporte detallado de una generación
        
        Args:
            generation_num: Número de generación
            top_n: Número de mejores individuos a mostrar
            show_all: Si True, muestra todos los individuos
        """
        if show_all:
            self.ga.print_all_individuals(generation_num)
        else:
            self.ga.print_generation_summary(generation_num, top_n)
    
    def print_best_evolution(self):
        """Imprime evolución de los mejores individuos por generación"""
        best_per_gen = self.ga.get_best_per_generation()
        
        print(f"\n{'='*80}")
        print("EVOLUCIÓN DE LOS MEJORES INDIVIDUOS POR GENERACIÓN")
        print(f"{'='*80}")
        print(f"{'Gen':<5} {'Fitness':<10} {'Parámetros'}")
        print("-" * 80)
        
        for best in best_per_gen[::5]:  # Mostrar cada 5 generaciones
            gen = best['generation']
            fitness = best['best_fitness']
            params = best['best_individual']
            print(f"{gen:<5} {fitness:<10.4f} {params}")
    
    def get_generation_details(self):
        """Retorna detalles completos de todas las generaciones"""
        return self.ga.get_generation_details()