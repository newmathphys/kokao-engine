# kokao_engine_v10_0/evolve_kokao.py
"""EvolveKokao — Эволюция архитектур через генетические алгоритмы."""

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kokao-v9")
)

from .core import KokaoCoreV9


class EvolveKokao:
    """Эволюция весов Kokao через генетический алгоритм.

    Fitness = 1 / (loss + ε)
    Операции: tournament selection, blend crossover, gaussian mutation
    """

    def __init__(
        self,
        n_features: int,
        population_size: int = 20,
        generations: int = 100,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.5,
    ):
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Инициализация популяции
        self.population = [
            [random.gauss(0, 1) for _ in range(n_features)] for _ in range(population_size)
        ]

        self.best_individual = None
        self.best_fitness = float("-inf")
        self.history = []

    def fitness(self, individual: list) -> float:
        """Fitness функция: 1 / (loss + ε)."""
        # ✅ Используем CPU для консистентности
        device = torch.device("cpu")
        core = KokaoCoreV9(weights=torch.tensor(individual, device=device))
        x = torch.randn(self.n_features, device=device)
        loss = core.train(x, target=100.0)
        return 1.0 / (loss + 1e-8)

    def select(self, tournament_size: int = 3) -> list:
        """Tournament selection."""
        contestants = random.sample(self.population, tournament_size)
        fitnesses = [self.fitness(ind) for ind in contestants]
        winner_idx = np.argmax(fitnesses)
        return contestants[winner_idx][:]

    def crossover(self, parent1: list, parent2: list) -> tuple:
        """Blend crossover (BLX-α)."""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        alpha = 0.5
        child1, child2 = [], []
        for g1, g2 in zip(parent1, parent2):
            low = min(g1, g2) - alpha * abs(g1 - g2)
            high = max(g1, g2) + alpha * abs(g1 - g2)
            child1.append(random.uniform(low, high))
            child2.append(random.uniform(low, high))
        return child1, child2

    def mutate(self, individual: list) -> list:
        """Gaussian mutation."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += random.gauss(0, 1)
        return individual

    def evolve(self, verbose: bool = True, tournament_size: int = 3) -> list:
        """Запустить эволюцию.

        Returns:
            best_individual: Лучший индивидуум

        """
        for gen in range(self.generations):
            # Evaluation
            fitnesses = [self.fitness(ind) for ind in self.population]

            # Statistics
            avg_fitness = np.mean(fitnesses)
            max_fitness = np.max(fitnesses)
            best_idx = np.argmax(fitnesses)

            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_individual = self.population[best_idx][:]

            self.history.append(
                {
                    "generation": gen,
                    "avg_fitness": avg_fitness,
                    "max_fitness": max_fitness,
                    "best_fitness": self.best_fitness,
                }
            )

            if verbose and gen % 10 == 0:
                print(
                    f"Gen {gen}: avg={avg_fitness:.4f}, max={max_fitness:.4f}, best={self.best_fitness:.4f}"
                )

            # Selection + Crossover + Mutation
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.select(tournament_size=tournament_size)
                parent2 = self.select(tournament_size=tournament_size)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            # Elitism: сохранить лучшего
            new_population[0] = self.population[best_idx][:]
            self.population = new_population

        return self.best_individual

    def get_best_core(self) -> KokaoCoreV9:
        """Получить лучшее ядро после эволюции."""
        if self.best_individual is None:
            self.evolve(verbose=False)
        return KokaoCoreV9(weights=torch.tensor(self.best_individual))
