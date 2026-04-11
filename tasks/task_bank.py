from dataclasses import dataclass
from typing import Optional
import os
import random
import asyncio
import builtins
import concurrent.futures


@dataclass
class Task:
    name: str
    difficulty: str
    prompt: str
    setup_files: dict
    grader: callable
    seed: int = 42
    step_budget: int = 20


# ---------------------------------------------------------------------------
# Sandboxed execution helpers
# ---------------------------------------------------------------------------

SAFED_BUILTINS = {
    k: v for k, v in builtins.__dict__.items()
    if k not in ('eval', 'exec', 'compile', '__import__', 'open', 'file', 'input')
}
SAFED_BUILTINS['print'] = lambda *args, **kwargs: None  # Suppress output

# Allow importing safe stdlib modules needed by typical solutions
_ALLOWED_MODULES = {
    'collections', 'dataclasses', 'typing', 'math', 'functools',
    'itertools', 'heapq', 'bisect', 'copy', 'enum', 'abc',
    'asyncio', 'time',
}

def _restricted_import(name, *args, **kwargs):
    if name.split('.')[0] in _ALLOWED_MODULES:
        return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)
    raise ImportError(f"Import of '{name}' is not allowed in grader sandbox")

SAFED_BUILTINS['__import__'] = _restricted_import


def _safe_exec(code: str, path: str, extra_globals: dict | None = None) -> dict:
    """Execute *code* in a restricted namespace and return the namespace."""
    namespace = {
        '__builtins__': SAFED_BUILTINS,
        '__name__': '__grader__',
        '__doc__': None,
    }
    if extra_globals:
        namespace.update(extra_globals)

    try:
        compiled = compile(code, path, 'exec')
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(exec, compiled, namespace)
            try:
                future.result(timeout=5)
            except concurrent.futures.TimeoutError:
                raise TimeoutError("Code execution timed out")
        return namespace
    except Exception as e:
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Grader 1 – debug_memory_leak  (Medium)
# ---------------------------------------------------------------------------

def grade_debug_memory_leak(workspace: str) -> float:
    target_path = os.path.join(workspace, "cache_manager.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, target_path, "exec")
    except SyntaxError:
        return 0.01

    import time
    ns = _safe_exec(code, target_path, extra_globals={"time": time})
    if "_error" in ns:
        return 0.01

    if "CacheManager" not in ns:
        return 0.1

    CacheManager = ns["CacheManager"]

    try:
        c = CacheManager(ttl_seconds=60)
        c.set("k1", "v1")
        # Time travel: make entry artificially expired
        c._cache["k1"]["ts"] = time.time() - 100
        
        # Test 1: get() returns None for expired entry
        check_get_expiry = (c.get("k1") is None)
        
        # Test 2: cleanup() removes them
        c.set("k2", "v2")
        c._cache["k2"]["ts"] = time.time() - 100
        c.cleanup()
        check_cleanup = ("k2" not in getattr(c, "_cache", {}))

        if check_get_expiry and check_cleanup:
            return 0.99
        if check_get_expiry and not check_cleanup:
            return 0.75
        if not check_get_expiry and check_cleanup:
            return 0.5
        return 0.25
    except Exception:
        return 0.25


# ---------------------------------------------------------------------------
# Grader 2 – fix_retry_logic  (Hard)
# ---------------------------------------------------------------------------

def grade_fix_retry_logic(workspace: str) -> float:
    target_path = os.path.join(workspace, "retry_utils.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, target_path, "exec")
    except SyntaxError:
        return 0.01

    # Mock time.sleep to run quickly and track delays
    class MockTime:
        def __init__(self):
            self.sleeps = []
        def sleep(self, seconds):
            self.sleeps.append(seconds)
    mock_time = MockTime()

    ns = _safe_exec(code, target_path, extra_globals={"time": mock_time})
    if "_error" in ns:
        return 0.01

    if "retry" not in ns or "RetryableError" not in ns:
        return 0.25

    retry = ns["retry"]
    RetryableError = ns["RetryableError"]

    bugs_fixed = 0

    # Test Bug 1: Delay should start at 1.0 (or default 1x)
    try:
        mock_time.sleeps.clear()
        @retry(max_attempts=2, backoff=2.0)
        def fail_fn():
            raise RetryableError("fail")
            
        try: fail_fn()
        except Exception: pass
        
        if len(mock_time.sleeps) > 0 and mock_time.sleeps[0] == 1.0:
            bugs_fixed += 1
    except Exception:
        pass

    # Test Bug 2: Catch only RetryableException (or allow original Exception to propagate immediately)
    try:
        calls = 0
        @retry(max_attempts=3, backoff=2.0)
        def fail_value():
            nonlocal calls
            calls += 1
            raise ValueError("fail")
            
        try: fail_value()
        except ValueError: pass
        except Exception: pass
            
        if calls == 1:
            bugs_fixed += 1
    except Exception:
        pass

    # Test Bug 3: Re-raise original exception on max retries, not swallow to return None
    try:
        @retry(max_attempts=2, backoff=2.0)
        def fail_retry():
            raise RetryableError("max fail")
            
        raised = False
        try:
            res = fail_retry()
        except RetryableError:
            raised = True
            
        if raised:
            bugs_fixed += 1
    except Exception:
        pass

    if bugs_fixed == 3: return 0.99
    if bugs_fixed == 2: return 0.75
    if bugs_fixed == 1: return 0.5
    return 0.25


# ---------------------------------------------------------------------------
# Grader 3 – implement_circuit_breaker  (Hard)
# ---------------------------------------------------------------------------

def grade_implement_circuit_breaker(workspace: str) -> float:
    target_path = os.path.join(workspace, "circuit_breaker.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, target_path, "exec")
    except SyntaxError:
        return 0.01

    import time
    ns = _safe_exec(code, target_path, extra_globals={"time": time})
    if "_error" in ns:
        return 0.01

    if "CircuitBreaker" not in ns:
        return 0.01

    CircuitBreaker = ns["CircuitBreaker"]
    CBOpen = ns.get("CircuitBreakerOpen", Exception)
    if not hasattr(CircuitBreaker, "call"):
        return 0.2

    score = 0.4
    try:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        def succeed(): return "ok"
        def fail(): raise ValueError("fail")

        # Test 1: Initially closed, executes OK
        r = cb.call(succeed)
        if r == "ok":
            score = 0.6

        # Test 2: Transitions to OPEN after threshold failures
        try: cb.call(fail) 
        except Exception: pass
        try: cb.call(fail)
        except Exception: pass
        
        is_open = False
        try:
            cb.call(succeed)
        except CBOpen:
            is_open = True
        except Exception:
            is_open = True # Credit if they throw anything stopping execution 

        if is_open:
            score = 0.8
            
        # Test 3: Transitions to HALF_OPEN after timeout
        time.sleep(0.15)
        is_half_open = False
        try:
            r = cb.call(succeed)
            if r == "ok":
                is_half_open = True
        except Exception:
            pass

        if is_open and is_half_open:
            score = 0.99

        return score
    except Exception:
        return score


# ---------------------------------------------------------------------------
# Grader 4 – fix_syntax_error  (Easy)
# ---------------------------------------------------------------------------

def grade_fix_syntax_error(workspace: str) -> float:
    target_path = os.path.join(workspace, "broken.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, "broken.py", "exec")
    except SyntaxError:
        return 0.01

    ns = _safe_exec(code, target_path)
    score = 0.01

    if "add" not in ns:
        score = 0.15
    else:
        try:
            if ns["add"](2, 3) == 5:
                score = 0.99
            else:
                score = 0.5
        except Exception:
            score = 0.5

    return min(max(score, 0.01), 0.99)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = [
    Task(
        name="fix_syntax_error",
        difficulty="Easy",
        prompt=(
            "Fix the syntax error in broken.py. The file contains \n"
            "a function with a missing colon after the def statement. \n"
            "The function should be named add(a, b) and return a + b."
        ),
        setup_files={
            "broken.py": (
                "def add(a, b)\n"
                "    return a + b\n"
            )
        },
        grader=grade_fix_syntax_error,
        seed=7,
        step_budget=5,
    ),
    Task(
        name="debug_memory_leak",
        difficulty="Medium",
        prompt=(
            "Fix the memory leak in cache_manager.py. The _cache dict "
            "grows unbounded because expired entries are never evicted. "
            "Add TTL-based expiration so entries older than ttl_seconds are "
            "removed on access and during cleanup()."
        ),
        setup_files={
            "cache_manager.py": (
                "import time\n"
                "\n"
                "class CacheManager:\n"
                "    def __init__(self, ttl_seconds=60):\n"
                "        self.ttl_seconds = ttl_seconds\n"
                "        self._cache = {}\n"
                "        \n"
                "    def set(self, key, value):\n"
                "        self._cache[key] = {\"value\": value, \"ts\": time.time()}\n"
                "        \n"
                "    def get(self, key):\n"
                "        if key in self._cache:\n"
                "            return self._cache[key][\"value\"]\n"
                "        return None\n"
                "        \n"
                "    def cleanup(self):\n"
                "        pass\n"
            ),
        },
        grader=grade_debug_memory_leak,
        seed=42,
        step_budget=15,
    ),
    Task(
        name="fix_retry_logic",
        difficulty="Hard",
        prompt=(
            "Fix the retry decorator in retry_utils.py. It has three bugs:\n"
            "1. It retries on ALL exceptions instead of only RetryableError\n"
            "2. The backoff multiplier is applied before the first retry (should start at 1x)\n"
            "3. It swallows the original exception — should re-raise after max retries"
        ),
        setup_files={
            "retry_utils.py": (
                "import time\n"
                "\n"
                "class RetryableError(Exception): pass\n"
                "\n"
                "def retry(max_attempts=3, backoff=2.0):\n"
                "    def decorator(func):\n"
                "        def wrapper(*args, **kwargs):\n"
                "            delay = backoff  # bug 1: should start at 1.0\n"
                "            for attempt in range(max_attempts):\n"
                "                try:\n"
                "                    return func(*args, **kwargs)\n"
                "                except Exception as e:  # bug 2: catches all exceptions\n"
                "                    if attempt == max_attempts - 1:\n"
                "                        return None  # bug 3: swallows exception\n"
                "                    time.sleep(delay)\n"
                "                    delay *= backoff\n"
                "        return wrapper\n"
                "    return decorator\n"
            ),
        },
        grader=grade_fix_retry_logic,
        seed=137,
        step_budget=20,
    ),
    Task(
        name="implement_circuit_breaker",
        difficulty="Hard",
        prompt=(
            "Implement a circuit breaker pattern in circuit_breaker.py.\n"
            "Requirements:\n"
            "- CircuitBreaker(failure_threshold=3, recovery_timeout=30)\n"
            "- States: CLOSED (normal), OPEN (failing, reject calls), HALF_OPEN (testing)\n"
            "- Transitions: CLOSED→OPEN after failure_threshold failures\n"
            "- Transitions: OPEN→HALF_OPEN after recovery_timeout seconds\n"
            "- Transitions: HALF_OPEN→CLOSED on success, HALF_OPEN→OPEN on failure\n"
            "- call(func, *args) method that enforces the circuit state\n"
            "- Raise CircuitBreakerOpen when circuit is OPEN"
        ),
        setup_files={},
        grader=grade_implement_circuit_breaker,
        seed=999,
        step_budget=25,
    ),
]


# ---------------------------------------------------------------------------
# TaskBank
# ---------------------------------------------------------------------------

class TaskBank:
    def __init__(self):
        self.tasks = TASKS
        self.current_task: Optional[Task] = None

    def sample(self, difficulty: Optional[str] = None, seed: Optional[int] = None) -> Task:
        rng = random.Random(seed if seed is not None else 42)
        if difficulty:
            filtered = [t for t in self.tasks if t.difficulty.lower() == difficulty.lower()]
            if filtered:
                self.current_task = rng.choice(filtered)
            else:
                self.current_task = rng.choice(self.tasks)
        else:
            self.current_task = rng.choice(self.tasks)
        return self.current_task

    def get_current(self) -> Optional[Task]:
        return self.current_task

    def get_by_name(self, name: str):
        for task in self.tasks:
            if task.name == name:
                self.current_task = task
                return task
        return None

    def grade(self, workspace: str) -> float:
        if not self.current_task:
            return 0.01
        raw_score = self.current_task.grader(workspace)
        return min(max(float(raw_score), 0.01), 0.99)

    def reset(self) -> dict:
        self.current_task = None
        return {"status": "reset", "task": None}


def get_task_bank() -> TaskBank:
    return TaskBank()