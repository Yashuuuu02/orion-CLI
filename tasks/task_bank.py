from dataclasses import dataclass
from typing import Optional
import os
import random
import asyncio
import builtins


@dataclass
class Task:
    name: str
    difficulty: str
    prompt: str
    setup_files: dict
    grader: callable
    seed: int = 42


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
    'asyncio',
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
        exec(compiled, namespace)
        return namespace
    except Exception as e:
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Grader 1 – debug_off_by_one  (Medium)
# ---------------------------------------------------------------------------

def grade_debug_off_by_one(workspace: str) -> float:
    """Grade the binary-search off-by-one fix."""
    target_path = os.path.join(workspace, "search.py")
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

    if "binary_search" not in ns:
        return 0.01

    fn = ns["binary_search"]

    # Define test cases: (arr, target, expected_index)
    test_cases = [
        ([1, 2, 3, 4, 5], 5, 4),    # last element — the original bug
        ([1, 2, 3], 1, 0),           # first element
        ([1], 1, 0),                 # single-element array
        ([1, 3, 5, 7, 9], 7, 3),     # middle element
        ([2, 4, 6], 5, -1),          # element not present
    ]

    passed = 0
    for arr, target, expected in test_cases:
        try:
            result = fn(arr, target)
            if not isinstance(result, int):
                continue
            if result == expected:
                passed += 1
        except Exception:
            # Function exists but throws — still counts as 0.1
            pass

    if passed == len(test_cases):
        return 0.99
    if passed >= 3:
        return 0.7
    if passed >= 1:
        return 0.4
    # Function exists but all tests fail or throw
    return 0.1


# ---------------------------------------------------------------------------
# Grader 2 – fix_race_condition  (Hard)
# ---------------------------------------------------------------------------

def grade_fix_race_condition(workspace: str) -> float:
    """Grade the asyncio race-condition fix."""
    target_path = os.path.join(workspace, "async_counter.py")
    if not os.path.exists(target_path):
        return 0.01

    try:
        code = open(target_path).read()
        compile(code, target_path, "exec")
    except SyntaxError:
        return 0.1

    # Check source text for locking primitives
    has_lock = ("asyncio.Lock" in code or "asyncio.Semaphore" in code
                or "Lock()" in code)

    # Execute with asyncio available so the submitted code can import it
    ns = _safe_exec(code, target_path, extra_globals={"asyncio": asyncio})
    if "_error" in ns:
        return 0.1

    if "Counter" not in ns:
        return 0.1

    CounterClass = ns["Counter"]

    # Concurrent correctness test
    async def _concurrent_test() -> bool:
        try:
            counter = CounterClass()
            tasks = [counter.increment() for _ in range(100)]
            await asyncio.gather(*tasks)
            return counter.value == 100
        except Exception:
            return False

    try:
        concurrent_ok = asyncio.get_event_loop().run_until_complete(_concurrent_test())
    except RuntimeError:
        # No running event loop — create a new one
        concurrent_ok = asyncio.run(_concurrent_test())

    if has_lock and concurrent_ok:
        return 0.99
    if has_lock:
        return 0.7
    # increment exists but no lock
    return 0.4


# ---------------------------------------------------------------------------
# Grader 3 – implement_lru_cache  (Hard)
# ---------------------------------------------------------------------------

def grade_implement_lru_cache(workspace: str) -> float:
    """Grade the LRU cache implementation."""
    target_path = os.path.join(workspace, "lru_cache.py")
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

    if "LRUCache" not in ns:
        return 0.01

    CacheClass = ns["LRUCache"]

    def _run_tests() -> int:
        passed = 0

        # Test 1: basic get/put
        try:
            c = CacheClass(2)
            c.put(1, 10)
            c.put(2, 20)
            if c.get(1) == 10 and c.get(2) == 20:
                passed += 1
        except Exception:
            pass

        # Test 2: eviction
        try:
            c = CacheClass(2)
            c.put(1, 10)
            c.put(2, 20)
            c.put(3, 30)  # evicts key 1
            if c.get(1) == -1 and c.get(3) == 30:
                passed += 1
        except Exception:
            pass

        # Test 3: update existing key
        try:
            c = CacheClass(2)
            c.put(1, 10)
            c.put(2, 20)
            c.put(1, 100)  # update, key 1 becomes most recent
            c.put(3, 30)   # evicts key 2 (LRU), not key 1
            if c.get(1) == 100 and c.get(2) == -1 and c.get(3) == 30:
                passed += 1
        except Exception:
            pass

        # Test 4: get refreshes recency
        try:
            c = CacheClass(2)
            c.put(1, 10)
            c.put(2, 20)
            c.get(1)       # key 1 is now most recently used
            c.put(3, 30)   # evicts key 2 (LRU)
            if c.get(1) == 10 and c.get(2) == -1:
                passed += 1
        except Exception:
            pass

        # Test 5: capacity 1
        try:
            c = CacheClass(1)
            c.put(1, 10)
            c.put(2, 20)  # evicts key 1
            if c.get(1) == -1 and c.get(2) == 20:
                passed += 1
        except Exception:
            pass

        return passed

    try:
        passed = _run_tests()
    except Exception:
        return 0.1

    if passed == 5:
        return 0.99
    if passed == 4:
        return 0.8
    if passed == 3:
        return 0.6
    if passed == 2:
        return 0.4
    if passed == 1:
        return 0.2
    return 0.1


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = [
    Task(
        name="debug_off_by_one",
        difficulty="Medium",
        prompt=(
            "Fix the binary search implementation in search.py. "
            "It has an off-by-one error that causes it to miss the last element "
            "in the array. The function signature is: "
            "def binary_search(arr, target) -> int"
        ),
        setup_files={
            "search.py": (
                "def binary_search(arr, target):\n"
                "    left, right = 0, len(arr) - 2  # bug: should be len(arr) - 1\n"
                "    while left <= right:\n"
                "        mid = (left + right) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            left = mid + 1\n"
                "        else:\n"
                "            right = mid - 1\n"
                "    return -1\n"
            ),
        },
        grader=grade_debug_off_by_one,
        seed=42,
    ),
    Task(
        name="fix_race_condition",
        difficulty="Hard",
        prompt=(
            "Fix the race condition in async_counter.py. The increment() "
            "method uses a read-modify-write pattern without locking, causing "
            "incorrect counts under concurrent access. Fix it to use asyncio.Lock."
        ),
        setup_files={
            "async_counter.py": (
                "import asyncio\n"
                "\n"
                "class Counter:\n"
                "    def __init__(self):\n"
                "        self.value = 0\n"
                "\n"
                "    async def increment(self):\n"
                "        current = self.value      # race condition: read\n"
                "        await asyncio.sleep(0)    # yield — another coroutine can run here\n"
                "        self.value = current + 1  # write stale value\n"
            ),
        },
        grader=grade_fix_race_condition,
        seed=137,
    ),
    Task(
        name="implement_lru_cache",
        difficulty="Hard",
        prompt=(
            "Implement an LRU cache in lru_cache.py with these requirements:\n"
            "- Class LRUCache(capacity: int)\n"
            "- get(key) -> int: return value or -1 if not found\n"
            "- put(key, value): insert or update, evict LRU if at capacity\n"
            "Both operations must run in O(1) time."
        ),
        setup_files={},
        grader=grade_implement_lru_cache,
        seed=999,
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

    def grade(self, workspace: str) -> float:
        if not self.current_task:
            return 0.01
        return self.current_task.grader(workspace)

    def reset(self) -> dict:
        self.current_task = None
        return {"status": "reset", "task": None}


def get_task_bank() -> TaskBank:
    return TaskBank()