"""Тесты для robustness (6 тестов)."""

import os
import sys

import pytest

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.robustness import log_execution, retry, rollback


class TestRetry:
    """@retry декоратор."""

    def test_retry_success_first(self):
        calls = [0]

        @retry(max_retries=3)
        def succeed():
            calls[0] += 1
            return "ok"

        assert succeed() == "ok"
        assert calls[0] == 1

    def test_retry_success_after_failures(self):
        calls = [0]

        @retry(max_retries=3, delay=0.01)
        def succeed_later():
            calls[0] += 1
            if calls[0] < 3:
                raise ValueError("temp")
            return "ok"

        assert succeed_later() == "ok"
        assert calls[0] == 3

    def test_retry_exhausted(self):
        calls = [0]

        @retry(max_retries=3, delay=0.01)
        def always_fail():
            calls[0] += 1
            raise ValueError("perm")

        with pytest.raises(ValueError):
            always_fail()
        assert calls[0] == 3


class TestRollback:
    """@rollback декоратор."""

    def test_rollback_on_failure(self):
        state = {"v": 0}

        def save():
            return state.copy()

        def restore(s):
            state.update(s)

        @rollback(snapshot_func=save, restore_func=restore)
        def fail_op():
            state["v"] = 100
            raise ValueError("fail")

        with pytest.raises(RuntimeError):
            fail_op()
        assert state["v"] == 0

    def test_no_rollback_on_success(self):
        state = {"v": 0}

        def save():
            return state.copy()

        def restore(s):
            state.update(s)

        @rollback(snapshot_func=save, restore_func=restore)
        def success_op():
            state["v"] = 100
            return "ok"

        assert success_op() == "ok"
        assert state["v"] == 100


class TestLogExecution:
    """@log_execution декоратор."""

    def test_log_exec(self):
        @log_execution
        def add(a, b):
            return a + b

        assert add(2, 3) == 5
