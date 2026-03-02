"""Evolution Generations — 15 тестов."""

import os
import sys

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.evolve import EvolveKokao


class TestEvolveGen:
    """Evolution поколения."""

    def test_evolve_one_generation(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=1)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_five_generations(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=5)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_ten_generations(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=10)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_twenty_generations(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=20)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_fifty_generations(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=50)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_hundred_generations(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=100)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_tiny_population(self):
        ev = EvolveKokao(n_features=5, population_size=2, generations=5)
        # tournament_size должен быть <= population_size
        best = ev.evolve(verbose=False, tournament_size=2)
        assert len(best) == 5

    def test_evolve_large_population(self):
        ev = EvolveKokao(n_features=5, population_size=100, generations=5)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_single_feature(self):
        ev = EvolveKokao(n_features=1, population_size=10, generations=5)
        best = ev.evolve(verbose=False)
        assert len(best) == 1

    def test_evolve_many_features(self):
        ev = EvolveKokao(n_features=50, population_size=10, generations=5)
        best = ev.evolve(verbose=False)
        assert len(best) == 50

    def test_evolve_zero_mutation(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=5, mutation_rate=0.0)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_high_mutation(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=5, mutation_rate=0.5)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_zero_crossover(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=5, crossover_rate=0.0)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_high_crossover(self):
        ev = EvolveKokao(n_features=5, population_size=10, generations=5, crossover_rate=1.0)
        best = ev.evolve(verbose=False)
        assert len(best) == 5

    def test_evolve_fitness_improvement(self):
        ev = EvolveKokao(n_features=5, population_size=20, generations=20)
        initial_fitness = max(ev.fitness(ind) for ind in ev.population)
        ev.evolve(verbose=False)
        assert ev.best_fitness >= initial_fitness
