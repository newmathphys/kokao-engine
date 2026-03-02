"""Robustness decorators для Kokao Engine."""

import functools
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Декоратор для повторных попыток выполнения функции.

    Args:
        max_retries: Максимальное количество попыток
        delay: Начальная задержка между попытками (сек)
        backoff: Множитель задержки (экспоненциальный backoff)

    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Попытка {attempt}/{max_retries} не удалась: {e}")

                    if attempt == max_retries:
                        raise

                    time.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception  # type: ignore

        return wrapper

    return decorator


def rollback(snapshot_func: Optional[Callable] = None, restore_func: Optional[Callable] = None):
    """Декоратор для отката изменений при ошибке.

    Args:
        snapshot_func: Функция для сохранения состояния (default: None)
        restore_func: Функция для восстановления состояния (default: None)

    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            snapshot = None

            # Сохраняем состояние
            if snapshot_func is not None:
                try:
                    snapshot = snapshot_func(*args, **kwargs)
                    logger.info(f"Состояние сохранено для {func.__name__}")
                except Exception as e:
                    logger.warning(f"Не удалось сохранить состояние: {e}")

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Откатываем состояние
                if restore_func is not None and snapshot is not None:
                    try:
                        restore_func(snapshot)
                        logger.info(f"Состояние откатчено для {func.__name__}")
                    except Exception as restore_error:
                        logger.error(f"Не удалось откатить состояние: {restore_error}")

                raise RuntimeError(f"Ошибка в {func.__name__}, откат выполнен: {e}")

        return wrapper

    return decorator


def log_execution(func: Callable) -> Callable:
    """Декоратор для логирования выполнения функции."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"Вызов {func.__name__} с args={args}, kwargs={kwargs}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} выполнен за {elapsed:.3f} сек")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} завершился ошибкой за {elapsed:.3f} сек: {e}")
            raise

    return wrapper
