"""AutoLogic — авто-генерация правил."""

import numpy as np
import torch

from .core import KokaoCoreV9


class LogicRule:
    """Правило: условие → действие."""

    def __init__(self, condition: str, action: str):
        self.condition = condition
        self.action = action

    def evaluate(self, S: float, w: torch.Tensor):
        if "S >" in self.condition and S > float(self.condition.split(">")[1]):
            return self._apply(w)
        if "S <" in self.condition and S < float(self.condition.split("<")[1]):
            return self._apply(w)
        return None

    def _apply(self, w: torch.Tensor):
        if "increase" in self.action:
            return w * 1.1
        if "decrease" in self.action:
            return w * 0.9
        return w


class AutoLogic:
    """Авто-генерация правил (Decision Trees)."""

    def __init__(self, core: KokaoCoreV9, max_depth: int = 3):
        self.core = core
        self.max_depth = max_depth

    def generate_rules(self, X: torch.Tensor, S_values: list, threshold: float = 100.0) -> list:
        """Генерация правил из данных.

        Args:
            X: Входные данные (n_samples, n_features)
            S_values: Значения сигнала для каждого примера
            threshold: Порог для классификации

        Returns:
            rules: Список LogicRule

        """
        if not HAS_SKLEARN:
            return []  # Mocked

        # Подготовка данных
        X_np = X.numpy() if isinstance(X, torch.Tensor) else np.array(X)
        y = (np.array(S_values) > threshold).astype(int)

        # Обучение дерева
        clf = DecisionTreeClassifier(max_depth=self.max_depth)
        clf.fit(X_np, y)

        # Извлечение правил (упрощённо)
        rules = self._extract_rules(clf, X_np.shape[1])
        return rules

    def _extract_rules(self, clf, n_features: int) -> list:
        """Извлечь правила из дерева решений."""
        rules = []
        # Упрощённая экстракция
        # В реальной реализации нужно парсить tree_ структуру
        rules.append(LogicRule("S > 100", "increase"))
        rules.append(LogicRule("S < 50", "decrease"))
        return rules
