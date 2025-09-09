import random
import numpy as np
from typing import List, Tuple, Callable

class GeneticAlgorithm:
    def __init__(self, population_size: int = 50, generations: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history_generation = []  # Fitness de cada generación
        self.fitness_history_global = []      # Mejor fitness acumulado
        self.fitness_avg_history = []
        self.generation_details = [] 
    
    def create_individual(self, param_ranges: dict) -> dict:
        """Crea un individuo aleatorio basado en los rangos de parámetros"""
        individual = {}
        for param, (min_val, max_val, param_type) in param_ranges.items():
            if param_type == 'int':
                individual[param] = random.randint(min_val, max_val)
            elif param_type == 'float':
                individual[param] = random.uniform(min_val, max_val)
            elif param_type == 'choice':
                individual[param] = random.choice(min_val)  # min_val contiene las opciones
        return individual
    
    def create_population(self, param_ranges: dict) -> List[dict]:
        """Crea la población inicial"""
        return [self.create_individual(param_ranges) for _ in range(self.population_size)]
    
    def tournament_selection(self, population: List[dict], fitness_scores: List[float], 
                           tournament_size: int = 3) -> dict:
        """Selección por torneo"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index].copy()
    
    def crossover(self, parent1: dict, parent2: dict, param_ranges: dict, alpha: float = 0.5) -> Tuple[dict, dict]:
        """Cruce específico según tipo de parámetro"""
        child1, child2 = parent1.copy(), parent2.copy()

        if random.random() < self.crossover_rate:
            for param, (min_val, max_val, param_type) in param_ranges.items():
                if param_type == 'int':
                    # One-point crossover estilo binario
                    if random.random() < 0.5:
                        child1[param], child2[param] = child2[param], child1[param]

                elif param_type == 'float':
                    # BLX-α crossover (mejor para valores reales)
                    p1, p2 = parent1[param], parent2[param]
                    low, high = min(p1, p2), max(p1, p2)
                    diff = high - low
                    new_low = max(min_val, low - alpha * diff)
                    new_high = min(max_val, high + alpha * diff)

                    child1[param] = random.uniform(new_low, new_high)
                    child2[param] = random.uniform(new_low, new_high)

                elif param_type == 'choice':
                    # Uniform crossover para categóricos
                    if random.random() < 0.5:
                        child1[param], child2[param] = child2[param], child1[param]

        return child1, child2
    
    def mutate(self, individual: dict, param_ranges: dict, generation: int = 0) -> dict:
        """Mutación adaptativa según la generación"""
        mutated = individual.copy()

        # Tasa de mutación adaptativa: inicia alta y baja hacia 0
        adapt_mut_rate = self.mutation_rate * (1 - generation / self.generations)

        for param, (min_val, max_val, param_type) in param_ranges.items():
            if random.random() < adapt_mut_rate:
                if param_type == 'int':  # parámetros discretos
                    mutated[param] = random.randint(min_val, max_val)

                elif param_type == 'float':  # parámetros reales
                    # Mutación gaussiana con desviación proporcional al rango
                    current_val = mutated[param]
                    mutation_range = (max_val - min_val) * 0.1
                    new_val = current_val + random.gauss(0, mutation_range)
                    mutated[param] = max(min_val, min(max_val, new_val))

                elif param_type == 'choice':  # parámetros categóricos
                    # Elegir otra categoría distinta a la actual
                    choices = [c for c in min_val if c != mutated[param]]
                    if choices:
                        mutated[param] = random.choice(choices)

        return mutated
    
    def evolve(self, param_ranges: dict, fitness_function: Callable) -> dict:
        """Proceso evolutivo principal"""
        population = self.create_population(param_ranges)
        
        for generation in range(self.generations):
            
            print(f"\n{'='*60}")
            print(f"GENERACIÓN {generation}")
            print(f"{'='*60}")
            
            fitness_scores = []
            for i, individual in enumerate(population):
                fitness = fitness_function(individual)
                fitness_scores.append(fitness)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
                
                print(f"Individuo {i}: {individual} - Fitness: {fitness:.4f}")
            
            #self.fitness_history.append(max(fitness_scores))
            self.fitness_history_generation.append(max(fitness_scores))
            self.fitness_history_global.append(self.best_fitness)
            
            avg_fitness = np.mean(fitness_scores)
            
            self.fitness_avg_history.append(avg_fitness)
            
            # Guardar detalles de la generación
            generation_info = {
                'generation': generation,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores),
                'min_fitness': min(fitness_scores),
                'max_fitness': max(fitness_scores),
                'best_individual': population[np.argmax(fitness_scores)],
                'individuals': []
            }

            for idx, ind in enumerate(population):
                generation_info['individuals'].append({
                    'individual_id': idx,
                    'params': ind,
                    'fitness': fitness_scores[idx]
                })

            self.generation_details.append(generation_info)

            print(f"Generación {generation}: Mejor fitness = {self.best_fitness:.4f}, "
                    f"Promedio = {avg_fitness:.4f}")
            
            new_population = []
            best_idx = np.argmax(fitness_scores)
            new_population.append(population[best_idx].copy())

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2, param_ranges)
                child1 = self.mutate(child1, param_ranges, generation)
                child2 = self.mutate(child2, param_ranges, generation)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return self.best_individual

    
    def get_generation_details(self):
        """Retorna detalles completos de todas las generaciones"""
        return self.generation_details
    
    def print_generation_summary(self, generation_num, top_n=5):
        """Imprime resumen de una generación específica"""
        if generation_num >= len(self.generation_details):
            print(f"Generación {generation_num} no encontrada")
            return
        
        gen_data = self.generation_details[generation_num]
        print(f"\n{'='*60}")
        print(f"GENERACIÓN {generation_num}")
        print(f"{'='*60}")
        print(f"Mejor fitness: {gen_data['best_fitness']:.4f}")
        print(f"Fitness promedio: {gen_data['avg_fitness']:.4f} ± {gen_data['std_fitness']:.4f}")
        print(f"Rango fitness: [{gen_data['min_fitness']:.4f}, {gen_data['max_fitness']:.4f}]")
        
        print(f"\nTOP {top_n} INDIVIDUOS:")
        print("-" * 80)
        for i, ind in enumerate(gen_data['individuals'][:top_n]):
            print(f"#{i+1} (ID: {ind['individual_id']}) - Fitness: {ind['fitness']:.4f}")
            print(f"    Parámetros: {ind['params']}")
    
    def print_all_individuals(self, generation_num):
        """Imprime todos los individuos de una generación"""
        if generation_num >= len(self.generation_details):
            print(f"Generación {generation_num} no encontrada")
            return
        
        gen_data = self.generation_details[generation_num]
        print(f"\n{'='*80}")
        print(f"TODOS LOS INDIVIDUOS - GENERACIÓN {generation_num}")
        print(f"{'='*80}")
        
        for i, ind in enumerate(gen_data['individuals']):
            print(f"Individuo {ind['individual_id']:2d} - Fitness: {ind['fitness']:.4f} - {ind['params']}")
    
    def get_best_per_generation(self):
        """Retorna lista con el mejor individuo de cada generación"""
        best_per_gen = []
        for gen_data in self.generation_details:
            best_per_gen.append({
                'generation': gen_data['generation'],
                'best_individual': gen_data['best_individual'],
                'best_fitness': gen_data['best_fitness']
            })
        return best_per_gen