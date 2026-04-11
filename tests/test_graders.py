import tempfile, os, pytest
from tasks.task_bank import grade_fix_syntax_error, grade_debug_memory_leak, grade_fix_retry_logic, grade_implement_circuit_breaker


def test_fix_syntax_error_perfect():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "broken.py"), "w").write("def add(a, b):\n    return a + b\n")
        assert grade_fix_syntax_error(d) == 0.99


def test_fix_syntax_error_empty():
    with tempfile.TemporaryDirectory() as d:
        assert grade_fix_syntax_error(d) == 0.01


def test_fix_syntax_error_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "broken.py"), "w").write("def add(a, b)\n    return a + b\n")
        assert grade_fix_syntax_error(d) == 0.01


def test_circuit_breaker_empty():
    with tempfile.TemporaryDirectory() as d:
        assert grade_implement_circuit_breaker(d) == 0.01


def test_retry_logic_empty():
    with tempfile.TemporaryDirectory() as d:
        assert grade_fix_retry_logic(d) == 0.01


def test_memory_leak_empty():
    with tempfile.TemporaryDirectory() as d:
        assert grade_debug_memory_leak(d) == 0.01
