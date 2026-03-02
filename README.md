# Kokao Engine v1.0 – Gradient Signal Inversion Engine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-384%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-success)](https://codecov.io/gh/newmathphys/kokao-engine)
[![GitHub](https://img.shields.io/badge/github-newmathphys/kokao--engine-blue)](https://github.com/newmathphys/kokao-engine)
[![PyPI version](https://img.shields.io/pypi/v/kokao-engine)](https://pypi.org/project/kokao-engine)
[![Documentation](https://img.shields.io/readthedocs/kokao-engine)](https://kokao-engine.readthedocs.io)
[![Hugging Face](https://img.shields.io/badge/🤗%20Space-newmathphys/kokao--demo-yellow)](https://huggingface.co/spaces/newmathphys/kokao-demo)

**Kokao Engine** – это градиентный движок инверсии сигналов, реализующий метод функционально-независимых структур (метод Косякова). Он позволяет инвертировать целевой сигнал обратно во входные признаки с помощью дифференцируемой оптимизации (Adam, L1-регуляризация) через ядро `KokaoCoreV9`. Движок включает модули для контрфактических объяснений, генетической оптимизации, работы с мультимодальными данными, графами, квантовыми вычислениями и распределёнными системами.

---

## 🧱 Основная идея

- **Ядро** `KokaoCoreV9` хранит нормализованный вектор весов `w` (`∑|w| = 100`). Метод `signal(x)` вычисляет `S = wᵀx`.
- **Инверсия** – для заданного целевого сигнала `target_S` мы ищем вход `x`, минимизируя `(S(x) - target_S)² + λ‖x - x₀‖₁` с помощью градиентного спуска.
- **Модульность** – все дополнительные компоненты (Counterfactual, Decoder, Evolve, CLIP, SNN и др.) реализованы как независимые расширения, не изменяющие ядро.

---

## ⚙️ Установка

```bash
pip install kokao-engine
# или с опциональными зависимостями:
pip install kokao-engine[snn,quantum,ray,all]
```

## 🚀 Быстрый старт

```python
import torch
from kokao.core import KokaoCoreV9
from kokao.counterfactual import CounterfactualKokao

# 1. Создание ядра
core = KokaoCoreV9(n_features=5)

# 2. Прямой сигнал
x = torch.randn(5)
print("Signal:", core.signal(x))

# 3. Обучение на целевой сигнал (классическое)
loss = core.train(x, target=100.0)  # loss возвращается, можно не использовать

# 4. Контрфактический анализ
cf = CounterfactualKokao(core)
x_cf = cf.counterfactual(x, target_delta=20.0)
print("Counterfactual signal:", core.signal(x_cf))
```

## 📦 Состав (модули и тесты)

| Модуль | Тестов | Статус | Краткое описание |
|--------|--------|--------|------------------|
| Core | 18 | ✅ | Ядро: веса, сигнал, обучение, забывание, autograd |
| LearnableForget | 12 | ✅ | Дифференцируемая скорость забывания |
| EvolveKokao | 15 | ✅ | Генетическая оптимизация весов |
| Counterfactual | 10 | ✅ | Контрфактические объяснения |
| Decoder | 8 | ✅ | Генерация входа через градиентный подъём |
| CLIP | 10 | ✅ | Мультимодальные эмбеддинги (текст + изображение) |
| SNN | 12 | ✅ | Спайковые нейронные сети (LIF) |
| KG | 10 | ✅ | Динамический граф знаний (Neo4j) |
| Quantum | 10 | ✅ | Квантовый бэкенд (Qiskit) |
| RL | 8 | ✅ | Обёртка для обучения с подкреплением |
| Dask | 8 | ✅ | Распределённые вычисления |
| AutoLogic | 10 | ✅ | Автоматическое извлечение логических правил |
| Ray (mock) | 8 | ✅ | Распределённое обучение (моки) |
| Hub API | 10 | ✅ | REST API для обмена эталонами |
| Integration | 20 | ✅ | Интеграционные тесты |
| E2E | 10 | ✅ | Сквозные сценарии |

**Всего активных тестов:** 384 (100% пройдены), 22 теста пропущены (опциональные зависимости).

## 📚 Документация

- **FULL_DESCRIPTION.md** – полное описание проекта.
- **ARCHITECTURE.md** – жизненный цикл тензора и поток данных.
- **MATH.md** – математическая основа.
- **MODULES_COMPLETE.md** – детальный список модулей.
- **CONTRIBUTING.md** – руководство для контрибьюторов.
- **CHANGELOG.md** – история версий.

## 📄 Лицензия и авторство

Проект распространяется под лицензией Apache 2.0. Полный текст в файле LICENSE.

**Авторы:** Виталий Калиновский, В. Овсейчик (2026).

Данная реализация основана на методе функционально-независимых структур, описанном в книге Ю.Б. Косякова (1999). Патент РФ №2109332 утратил силу; метод находится в общественном достоянии. Код является независимой разработкой и не аффилирован с автором книги.

## 🌐 Ссылки

- **GitHub:** github.com/newmathphys/kokao-engine
- **PyPI:** pypi.org/project/kokao-engine
- **Документация:** kokao-engine.readthedocs.io
- **Демо:** huggingface.co/spaces/newmathphys/kokao-demo
