"""Robustness Retry/Rollback — 8 тестов."""

import os
import sys
import time

import pytest

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.robustness import log_execution, retry, rollback


class TestRobustnessRetry:
    """@retry декоратор."""

    def test_retry_three_attempts(self):
        calls = [0]

        @retry(max_retries=3, delay=0.01)
        def fail_twice():
            calls[0] += 1
            if calls[0] < 3:
                raise ValueError("temp")
            return "ok"

        assert fail_twice() == "ok"
        assert calls[0] == 3

    def test_retry_exhausted_raises(self):
        calls = [0]

        @retry(max_retries=3, delay=0.01)
        def always_fail():
            calls[0] += 1
            raise ValueError("perm")

        with pytest.raises(ValueError):
            always_fail()
        assert calls[0] == 3

    def test_retry_backoff(self):
        times = []

        @retry(max_retries=3, delay=0.01, backoff=2.0)
        def slow_fail():
            times.append(time.time())
            raise ValueError("fail")

        with pytest.raises(ValueError):
            slow_fail()

        if len(times) >= 2:
            assert times[2] - times[1] > times[1] - times[0]

    def test_retry_success_first(self):
        @retry(max_retries=3)
        def succeed():
            return "ok"

        assert succeed() == "ok"


class TestRobustnessRollback:
    """@rollback декоратор."""

    def test_rollback_called_on_error(self):
        state = {"v": 0}
        rollback_called = [False]

        def save():
            return state.copy()

        def restore(s):
            state.update(s)
            rollback_called[0] = True

        @rollback(snapshot_func=save, restore_func=restore)
        def fail_op():
            state["v"] = 100
            raise ValueError("fail")

        with pytest.raises(RuntimeError):
            fail_op()

        assert rollback_called[0] is True
        assert state["v"] == 0

    def test_rollback_not_called_on_success(self):
        state = {"v": 0}
        rollback_called = [False]

        def save():
            return state.copy()

        def restore(s):
            rollback_called[0] = True

        @rollback(snapshot_func=save, restore_func=restore)
        def success_op():
            state["v"] = 100
            return "ok"

        assert success_op() == "ok"
        assert rollback_called[0] is False


class TestRobustnessLogExec:
    """@log_execution декоратор."""

    def test_log_exec_returns_result(self):
        @log_execution
        def add(a, b):
            return a + b

        assert add(2, 3) == 5
