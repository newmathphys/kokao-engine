# 📤 Публикация Kokao Engine на PyPI

## ✅ Пре-релиз чек-лист

### Файлы исправлены:
- [x] `pyproject.toml` — версия 1.0.0, лицензия Apache-2.0, URL на newmathphys
- [x] `VERSION` — синхронизирован с pyproject.toml
- [x] `LICENSE` — Apache 2.0
- [x] `.gitignore` — добавлены *.log, *.zip, кэши
- [x] `MANIFEST.in` — включает документацию и тесты
- [x] Удалены временные файлы (логи, кэши, артефакты)

### Тесты:
- [x] Все тесты проходят (406 тестов)
- [x] Сборка пакета успешна

---

## 🔑 Шаг 1: Создание токена PyPI

1. Зарегистрируйтесь на https://pypi.org/
2. Перейдите в **Account settings → API tokens**
3. Нажмите **Add API token**
4. Выберите:
   - **Token name**: `kokao-engine-publish`
   - **Scope**: `All projects` (или только `kokao-engine`)
5. Скопируйте токен (начинается с `pypi-`)

> ⚠️ Токен показывается только один раз! Сохраните его в надёжном месте.

---

## 🔐 Шаг 2: Настройка .pypirc

Создайте файл `~/.pypirc` с содержимым:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-ваш_токен_здесь

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-ваш_тестовый_токен_здесь
```

**Рекомендация**: Сначала протестируйте на TestPyPI!

---

## 📦 Шаг 3: Сборка пакета

```bash
cd /home/mrbritneyxxx/ДвижекKOKAO/kokao-engine

# Очистка старых сборок
rm -rf dist/ build/ src/*.egg-info

# Установка инструмента сборки
pip install build twine

# Сборка
python -m build
```

Проверьте, что созданы файлы:
- `dist/kokao_engine-1.0.0.tar.gz`
- `dist/kokao_engine-1.0.0-py3-none-any.whl`

---

## 🧪 Шаг 4: Тестирование на TestPyPI (рекомендуется)

```bash
# Загрузка на тестовый сервер
twine upload --repository testpypi dist/*

# Проверка (в новом venv!)
python -m pip install --index-url https://test.pypi.org/simple/ kokao-engine
```

---

## 🚀 Шаг 5: Публикация на PyPI

```bash
# Загрузка
twine upload dist/*

# Проверка
pip install kokao-engine
```

---

## 🔍 Проверка после публикации

1. Откройте https://pypi.org/project/kokao-engine/
2. Проверьте:
   - Версию (1.0.0)
   - Описание (README)
   - Лицензию (Apache 2.0)
   - Ссылки (GitHub, Issues)
   - Файлы для скачивания

---

## 📝 Примечания

- **Повторная публикация той же версии невозможна**. Если нужно исправить — увеличьте версию (1.0.1).
- **Имя пакета**: `kokao-engine` (с дефисом)
- **Имя для импорта**: `kokao` (без дефиса)

---

## 🆘 Если что-то пошло не так

| Проблема | Решение |
|----------|---------|
| `Upload failed: 400` | Проверьте токен, имя пакета |
| `File already exists` | Увеличьте версию в pyproject.toml |
| `Invalid metadata` | Запустите `twine check dist/*` |
