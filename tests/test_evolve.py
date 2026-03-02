"""Evolution Tests (15 тестов)."""

import os
import sys

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.evolve import EvolveKokao


class TestEvolveKokao:
    """EvolveKokao тесты."""

    def test_init(self):
        ev = EvolveKokao(n_features=5, population_size=10)
        assert ev.n_features == 5
        assert len(ev.population) == 10

    def test_fitness(self):
        ev = EvolveKokao(n_features=5)
        individual = [1.0] * 5
        fitness = ev.fitness(individual)
        assert fitness > 0

    def test_select(self):
        ev = EvolveKokao(n_features=5, population_size=10)
        selected = ev.select(tournament_size=3)
        assert len(selected) == 5

    def test_crossover(self):
        ev = EvolveKokao(n_features=5)
        p1 = [1.0] * 5
        p2 = [2.0] * 5
        c1, c2 = ev.crossover(p1, p2)
        assert len(c1) == 5
        assert len(c2) == 5

    def test_mutate(self):
        ev = EvolveKokao(n_features=5)
        individual = [1.0] * 5
        mutated = ev.mutate(individual)
        assert len(mutated) == 5

    def test_evolve_basic(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=5)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_improves_fitness(self):
        ev = EvolveKokao(n_features=5, population_size=20, generations=10)
        initial_best_fitness = max(ev.fitness(ind) for ind in ev.population)
        ev.evolve(verbose=False)
        final_best_fitness = ev.best_fitness
        assert final_best_fitness >= initial_best_fitness

    def test_evolve_population_diversity(self):
        ev = EvolveKokao(n_features=5, population_size=20, generations=5)
        ev.evolve(verbose=False)
        fitnesses = [ev.fitness(ind) for ind in ev.population]
        assert max(fitnesses) - min(fitnesses) > 0

    def test_tournament_selection(self):
        ev = EvolveKokao(n_features=5, population_size=20)
        winners = []
        for _ in range(10):
            winner = ev.select(tournament_size=5)
            winners.append(ev.fitness(winner))
        assert all(w > 0 for w in winners)

    def test_crossover_rate(self):
        ev = EvolveKokao(n_features=5, crossover_rate=0.0)
        p1 = [1.0] * 5
        p2 = [2.0] * 5
        c1, c2 = ev.crossover(p1, p2)
        assert c1 == p1
        assert c2 == p2

    def test_mutation_rate(self):
        ev = EvolveKokao(n_features=5, mutation_rate=0.0)
        individual = [1.0] * 5
        mutated = ev.mutate(individual)
        assert mutated == individual

    def test_elitism(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=5)
        ev.evolve(verbose=False)
        assert ev.best_individual is not None

    def test_get_best_core(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=5)
        ev.evolve(verbose=False)
        core = ev.get_best_core()
        assert core is not None
        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_large_population(self):
        ev = EvolveKokao(n_features=5, population_size=100, generations=10)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_many_generations(self):
        ev = EvolveKokao(n_features=5, population_size=20, generations=50)
        best = ev.evolve(verbose=False)
        assert ev.best_fitness > 0
