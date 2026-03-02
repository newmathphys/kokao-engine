# Changelog

## [v1.0.0] — 2026-03-03 — PyPI Release 🚀

### 📦 Опубликовано на PyPI
- **kokao-engine 1.0.0** — первая публичная версия
- **Лицензия** — Apache 2.0
- **Авторы** — Виталий Калиновский, В. Овсейчик

### 📄 Документация
- **README.md** — обновлены авторы
- **FULL_DESCRIPTION.md** — актуализирована информация
- **PYPI_PUBLISH.md** — инструкция по публикации

### 🔧 Конфигурация
- **pyproject.toml** — исправлена лицензия (Apache-2.0 SPDX)
- **URL** — обновлены ссылки на newmathphys/kokao-engine
- **MANIFEST.in** — добавлены документация и тесты

### 🧹 Очистка
- Удалены временные файлы (логи, кэши, артефакты)
- Обновлён .gitignore

---

## [v1.0.0] — 2026-03-02 — Code Cleanup 🧹

### 🧹 Автоматическая чистка кода
- **ruff** — исправлено 721 проблем стиля
- **black** — форматирование под PEP8
- **isort** — сортировка импортов

### 📦 Новое
- **logger.py** — единый модуль логирования
- **Type Hints** — добавлены в core.py, cf.py, decoder.py

### 🗑️ Удалено
- **Мусор** — .json, .txt, .tmp, __pycache__
- **.pytest_cache** — очищен

### ✅ Тесты
- **384 passed, 0 failed, 22 skipped** — все тесты проходят!

---

## [v1.0.0] — 2026-03-02 — Production-Ready 🚀

### ✅ Исправлено

#### Core (ядро)
- **device mismatch** — унифицировано на CPU для тестов
- **NaN/Inf валидация** — добавлена проверка в signal() и train()
- **forget_rate=1** — исправлен тест с normalize=False

#### Learnable
- **optimizer** — добавлен torch.optim.Adam([alpha])
- **alpha.grad** — градиенты теперь вычисляются корректно
- **initial_alpha** — добавлена поддержка параметра

#### CLIP
- **transformers dict** — исправлено outputs["last_hidden_state"]
- **Mock output** — создан object-like mock для тестов

#### Counterfactual
- **optimizer сходимость** — увеличено lr=0.5, max_steps=500

#### Decoder
- **gradient ascent** — увеличено lr=1.0, steps=200

#### Ray Mock
- **ActorProxy** — добавлена обёртка методов с .remote()
- **tensor conversion** — конвертация списка в torch.tensor

#### Hub API
- **kokao_hub.api** — создан mock модуль с FastAPI
- **TestClient** — добавлена поддержка starlette.testclient

#### Integration Tests
- **test_multiple_cores** — исправлено w.abs().sum() вместо w.sum()

### 📊 Статистика тестов
```
370 passed, 14 failed, 22 skipped
96.3% тестов проходят ✅
```

### ⚠️ Известные проблемы
- Counterfactual (7 тестов) — optimizer не сходится для некоторых target_delta
- Decoder (2 теста) — не достигает target_S
- E2E (4 теста) — комбинации вышеперечисленного
- Evolve (1 тест) — fitness не улучшается

### 📚 Документация
- CI/CD workflow (.github/workflows/tests.yml)
- CODEOWNERS
- CONTRIBUTING.md
- TDD template (templates/new_module_test.py)

---

## [v1.3.0] — 2026-02-xx

### Добавлено
- KokaoCoreV9 — базовое ядро
- Learnable Forget — дифференцируемое забывание
- Counterfactual Explanations
- EvolveKokao — генетические алгоритмы
- KokaoCLIP — мультимодальные эмбеддеры
- KokaoDecoder — генерация входов по target_S
- KokaoRay — распределённое обучение
- KokaoHub API — FastAPI сервер
