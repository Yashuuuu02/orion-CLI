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
# Grader 1 – fix_tenacity_retry (Medium)
# ---------------------------------------------------------------------------

def grade_fix_tenacity_retry(workspace: str) -> float:
    target_path = os.path.join(workspace, "retry_utils.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, target_path, "exec")
    except SyntaxError:
        return 0.01

    ns = _safe_exec(code, target_path)
    if "_error" in ns:
        return 0.01

    if "TenacityRetry" not in ns or "RetryError" not in ns:
        return 0.25

    TenacityRetry = ns["TenacityRetry"]
    RetryError = ns["RetryError"]
    
    raises_retry_error = False
    skips_non_retryable = False
    returns_none = False
    
    # Test bug 2: raises RetryError instead of None
    try:
        @TenacityRetry(stop_max_attempt=2, wait_fixed=0.01)
        def always_fail():
            raise ValueError("fail")
            
        res = always_fail()
        if res is None:
            returns_none = True
    except RetryError:
        raises_retry_error = True
    except Exception:
        pass
        
    # Test bug 1: checks retry_on_exception
    non_retryable_calls = 0
    try:
        @TenacityRetry(stop_max_attempt=3, wait_fixed=0.01, retry_on_exception=lambda e: isinstance(e, KeyError))
        def fail_value():
            nonlocal non_retryable_calls
            non_retryable_calls += 1
            raise ValueError("fail")
            
        fail_value()
    except ValueError:
        if non_retryable_calls == 1:
            skips_non_retryable = True
    except Exception:
        pass

    if raises_retry_error and skips_non_retryable:
        return 0.99
    if raises_retry_error and not skips_non_retryable:
        return 0.75
    if returns_none and skips_non_retryable:
        return 0.50
    return 0.25


# ---------------------------------------------------------------------------
# Grader 2 – fix_cachetools_ttl (Medium)
# ---------------------------------------------------------------------------

def grade_fix_cachetools_ttl(workspace: str) -> float:
    target_path = os.path.join(workspace, "cache_manager.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, target_path, "exec")
    except SyntaxError:
        return 0.01

    ns = _safe_exec(code, target_path)
    if "_error" in ns:
        return 0.01

    if "TTLCache" not in ns:
        return 0.25

    TTLCache = ns["TTLCache"]
    
    getitem_checks_ttl = False
    expire_works = False
    
    import time
    try:
        cache = TTLCache(maxsize=10, ttl=0.1)
        cache["k1"] = "v1"
        time.sleep(0.15)
        
        # Check __getitem__ raises KeyError
        try:
            val = cache["k1"]
        except KeyError:
            getitem_checks_ttl = True
        except Exception:
            pass
            
        # Check expire() removes key
        cache["k2"] = "v2"
        time.sleep(0.15)
        cache.expire()
        
        if "k2" not in getattr(cache, "_cache", {}):
            expire_works = True
            
    except Exception:
        pass
        
    if getitem_checks_ttl and expire_works:
        return 0.99
    if getitem_checks_ttl and not expire_works:
        return 0.75
    if expire_works and not getitem_checks_ttl:
        return 0.50
    return 0.25


# ---------------------------------------------------------------------------
# Grader 3 – implement_pybreaker (Hard)
# ---------------------------------------------------------------------------

def grade_implement_pybreaker(workspace: str) -> float:
    target_path = os.path.join(workspace, "circuit_breaker.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, target_path, "exec")
    except SyntaxError:
        return 0.01

    ns = _safe_exec(code, target_path)
    if "_error" in ns:
        return 0.01

    if "CircuitBreaker" not in ns:
        return 0.01

    CircuitBreaker = ns["CircuitBreaker"]
    CBOpen = ns.get("CircuitBreakerError", Exception)
    if not hasattr(CircuitBreaker, "call"):
        return 0.20

    score = 0.40
    import time
    try:
        cb = CircuitBreaker(fail_max=2, reset_timeout=0.1)
        
        def succeed(): return "ok"
        def fail(): raise ValueError("fail")

        r = cb.call(succeed)
        if r == "ok":
            score = 0.60

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
            is_open = True 

        if is_open:
            score = 0.80
            
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
# Grader 4 – fix_async_race (Hard)
# ---------------------------------------------------------------------------

def grade_fix_async_race(workspace: str) -> float:
    target_path = os.path.join(workspace, "async_worker.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, target_path, "exec")
    except SyntaxError:
        return 0.01

    ns = _safe_exec(code, target_path)
    if "_error" in ns:
        return 0.01

    if "SharedCounter" not in ns:
        return 0.25

    SharedCounter = ns["SharedCounter"]
    
    import asyncio
    async def run_test():
        try:
            counter = SharedCounter()
            tasks = [counter.increment() for _ in range(100)]
            await asyncio.gather(*tasks)
            return counter.count
        except Exception:
            return -1

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_count = loop.run_until_complete(run_test())
        loop.close()
    except Exception:
        return 0.50
        
    if final_count == 100:
        return 0.99
    elif final_count > 0:
        return 0.50
    return 0.25


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = [
    Task(
        name="fix_tenacity_retry",
        difficulty="Medium",
        prompt=(
            "Fix the retry decorator in retry_utils.py. It has two bugs:\n"
            "1. It does not check retry_on_exception before retrying.\n"
            "2. It returns None instead of raising RetryError after max attempts."
        ),
        setup_files={
            "retry_utils.py": (
                "import time\n"
                "import functools\n"
                "\n"
                "class RetryError(Exception):\n"
                "    \"\"\"Raised when all retry attempts are exhausted.\"\"\"\n"
                "    pass\n"
                "\n"
                "class TenacityRetry:\n"
                "    \"\"\"Simplified tenacity-style retry decorator.\"\"\"\n"
                "    \n"
                "    def __init__(self, stop_max_attempt=3, wait_fixed=1.0, \n"
                "                 retry_on_exception=None):\n"
                "        self.stop_max_attempt = stop_max_attempt\n"
                "        self.wait_fixed = wait_fixed\n"
                "        self.retry_on_exception = retry_on_exception or (lambda e: True)\n"
                "        self.statistics = {\"attempt_number\": 0}\n"
                "    \n"
                "    def __call__(self, func):\n"
                "        @functools.wraps(func)\n"
                "        def wrapper(*args, **kwargs):\n"
                "            attempt = 0\n"
                "            while True:\n"
                "                attempt += 1\n"
                "                self.statistics[\"attempt_number\"] = attempt\n"
                "                try:\n"
                "                    return func(*args, **kwargs)\n"
                "                except Exception as e:\n"
                "                    # BUG 1: Should check retry_on_exception before retrying\n"
                "                    # BUG 2: Should raise RetryError after max attempts, not return None\n"
                "                    if attempt >= self.stop_max_attempt:\n"
                "                        return None  # Bug: should raise RetryError(e)\n"
                "                    time.sleep(self.wait_fixed)\n"
                "        return wrapper\n"
            )
        },
        grader=grade_fix_tenacity_retry,
        seed=42,
        step_budget=15,
    ),
    Task(
        name="fix_cachetools_ttl",
        difficulty="Medium",
        prompt=(
            "Fix the TTL cache in cache_manager.py. It has two bugs:\n"
            "1. `__getitem__` never checks if the entry has expired.\n"
            "2. `expire()` is broken/empty but should remove expired entries."
        ),
        setup_files={
            "cache_manager.py": (
                "import time\n"
                "from collections import OrderedDict\n"
                "\n"
                "class TTLCache:\n"
                "    \"\"\"Simplified cachetools-style TTL cache.\"\"\"\n"
                "    \n"
                "    def __init__(self, maxsize, ttl):\n"
                "        self.maxsize = maxsize\n"
                "        self.ttl = ttl\n"
                "        self._cache = OrderedDict()\n"
                "        self._timestamps = {}\n"
                "    \n"
                "    def __setitem__(self, key, value):\n"
                "        if len(self._cache) >= self.maxsize:\n"
                "            self._cache.popitem(last=False)\n"
                "        self._cache[key] = value\n"
                "        self._timestamps[key] = time.monotonic()\n"
                "    \n"
                "    def __getitem__(self, key):\n"
                "        # BUG: Never checks if entry has expired\n"
                "        if key not in self._cache:\n"
                "            raise KeyError(key)\n"
                "        return self._cache[key]  # Bug: should check TTL\n"
                "    \n"
                "    def expire(self):\n"
                "        # BUG: Never called, and even if called, doesn't work\n"
                "        pass  # Bug: should remove expired entries\n"
            )
        },
        grader=grade_fix_cachetools_ttl,
        seed=137,
        step_budget=20,
    ),
    Task(
        name="implement_pybreaker",
        difficulty="Hard",
        prompt=(
            "Implement a circuit breaker compatible with the pybreaker \n"
            "library interface. The CircuitBreaker class must implement:\n"
            "- __init__(fail_max=5, reset_timeout=60)\n"
            "- call(func, *args, **kwargs) — execute func, track failures\n"
            "- State machine: CLOSED → OPEN → HALF_OPEN\n"
            "- Raise CircuitBreakerError when OPEN\n"
            "- Use time.monotonic() for timeout tracking\n"
            "This pattern is used in production services to prevent cascade failures."
        ),
        setup_files={
            "circuit_breaker.py": (
                "import time\n"
                "\n"
                "class CircuitBreakerError(Exception):\n"
                "    pass\n"
                "\n"
                "class CircuitBreaker:\n"
                "    def __init__(self, fail_max=5, reset_timeout=60):\n"
                "        self.fail_max = fail_max\n"
                "        self.reset_timeout = reset_timeout\n"
                "        # TODO: Implement state machine\n"
            )
        },
        grader=grade_implement_pybreaker,
        seed=999,
        step_budget=25,
    ),
    Task(
        name="fix_async_race",
        difficulty="Hard",
        prompt=(
            "Fix the race condition in async_worker.py. This pattern \n"
            "appears in production async services. The SharedCounter.increment() \n"
            "method has a read-modify-write race condition because asyncio.sleep(0) \n"
            "yields control between the read and write operations."
        ),
        setup_files={
            "async_worker.py": (
                "import asyncio\n"
                "\n"
                "class SharedCounter:\n"
                "    def __init__(self):\n"
                "        self.count = 0\n"
                "        \n"
                "    async def increment(self):\n"
                "        # BUG: yield between read and write causes race condition in async environments\n"
                "        current = self.count\n"
                "        await asyncio.sleep(0)\n"
                "        self.count = current + 1\n"
            )
        },
        grader=grade_fix_async_race,
        seed=777,
        step_budget=30,
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