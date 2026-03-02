# Contributing to Kokao Engine

## Code Review Requirements

- ✅ Каждый PR должен быть проверен хотя бы одним другим разработчиком
- ✅ Проверка: логика, API, тесты, производительность
- ✅ Обязательные тесты должны проходить (95%+)
- ✅ Code Review checklist:
  - Логика алгоритма верна?
  - API соответствует стандартам?
  - Тесты покрывают edge cases?
  - Код документирован?

## Pull Request Checklist

- [ ] Тесты проходят (pytest tests/)
- [ ] Code Review одобрен
- [ ] Документация обновлена
- [ ] CHANGELOG.md обновлён

## TDD (Test-Driven Development)

- Писать тесты ПЕРВЫМИ
- Тесты должны покрывать edge cases (NaN, Inf, extreme values)
