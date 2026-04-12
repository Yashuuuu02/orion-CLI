"""Golden trajectory tests for all 4 current graders."""
import tempfile, os, pytest
from tasks.task_bank import (
    grade_fix_tenacity_retry,
    grade_fix_cachetools_ttl,
    grade_implement_pybreaker,
    grade_fix_async_race,
)


# ---------------------------------------------------------------------------
# fix_tenacity_retry
# ---------------------------------------------------------------------------

def test_tenacity_empty():
    with tempfile.TemporaryDirectory() as d:
        assert grade_fix_tenacity_retry(d) == 0.01


def test_tenacity_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "retry_utils.py"), "w").write("def broken(\n")
        assert grade_fix_tenacity_retry(d) == 0.01


def test_tenacity_class_exists_but_buggy():
    """Both bugs still present → 0.25."""
    code = (
        "import time, functools\n"
        "class RetryError(Exception): pass\n"
        "class TenacityRetry:\n"
        "    def __init__(self, stop_max_attempt=3, wait_fixed=1.0, retry_on_exception=None):\n"
        "        self.stop_max_attempt = stop_max_attempt\n"
        "        self.wait_fixed = wait_fixed\n"
        "        self.retry_on_exception = retry_on_exception or (lambda e: True)\n"
        "        self.statistics = {'attempt_number': 0}\n"
        "    def __call__(self, func):\n"
        "        @functools.wraps(func)\n"
        "        def wrapper(*args, **kwargs):\n"
        "            attempt = 0\n"
        "            while True:\n"
        "                attempt += 1\n"
        "                self.statistics['attempt_number'] = attempt\n"
        "                try:\n"
        "                    return func(*args, **kwargs)\n"
        "                except Exception as e:\n"
        "                    if attempt >= self.stop_max_attempt:\n"
        "                        return None\n"
        "                    time.sleep(self.wait_fixed)\n"
        "        return wrapper\n"
    )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "retry_utils.py"), "w").write(code)
        assert grade_fix_tenacity_retry(d) == 0.25


def test_tenacity_perfect():
    """Both bugs fixed → 0.99."""
    code = (
        "import time, functools\n"
        "class RetryError(Exception): pass\n"
        "class TenacityRetry:\n"
        "    def __init__(self, stop_max_attempt=3, wait_fixed=0.01, retry_on_exception=None):\n"
        "        self.stop_max_attempt = stop_max_attempt\n"
        "        self.wait_fixed = wait_fixed\n"
        "        self.retry_on_exception = retry_on_exception or (lambda e: True)\n"
        "        self.statistics = {'attempt_number': 0}\n"
        "    def __call__(self, func):\n"
        "        @functools.wraps(func)\n"
        "        def wrapper(*args, **kwargs):\n"
        "            attempt = 0\n"
        "            while True:\n"
        "                attempt += 1\n"
        "                self.statistics['attempt_number'] = attempt\n"
        "                try:\n"
        "                    return func(*args, **kwargs)\n"
        "                except Exception as e:\n"
        "                    if not self.retry_on_exception(e):\n"
        "                        raise\n"
        "                    if attempt >= self.stop_max_attempt:\n"
        "                        raise RetryError(str(e))\n"
        "                    time.sleep(self.wait_fixed)\n"
        "        return wrapper\n"
    )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "retry_utils.py"), "w").write(code)
        assert grade_fix_tenacity_retry(d) == 0.99


# ---------------------------------------------------------------------------
# fix_cachetools_ttl
# ---------------------------------------------------------------------------

def test_cachetools_empty():
    with tempfile.TemporaryDirectory() as d:
        assert grade_fix_cachetools_ttl(d) == 0.01


def test_cachetools_class_exists_but_buggy():
    """Both bugs still present → 0.25."""
    code = (
        "import time\n"
        "from collections import OrderedDict\n"
        "class TTLCache:\n"
        "    def __init__(self, maxsize, ttl):\n"
        "        self.maxsize = maxsize\n"
        "        self.ttl = ttl\n"
        "        self._cache = OrderedDict()\n"
        "        self._timestamps = {}\n"
        "    def __setitem__(self, key, value):\n"
        "        if len(self._cache) >= self.maxsize:\n"
        "            self._cache.popitem(last=False)\n"
        "        self._cache[key] = value\n"
        "        self._timestamps[key] = time.monotonic()\n"
        "    def __getitem__(self, key):\n"
        "        if key not in self._cache:\n"
        "            raise KeyError(key)\n"
        "        return self._cache[key]\n"
        "    def expire(self):\n"
        "        pass\n"
    )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "cache_manager.py"), "w").write(code)
        assert grade_fix_cachetools_ttl(d) == 0.25


def test_cachetools_perfect():
    """Both bugs fixed → 0.99."""
    code = (
        "import time\n"
        "from collections import OrderedDict\n"
        "class TTLCache:\n"
        "    def __init__(self, maxsize, ttl):\n"
        "        self.maxsize = maxsize\n"
        "        self.ttl = ttl\n"
        "        self._cache = OrderedDict()\n"
        "        self._timestamps = {}\n"
        "    def __setitem__(self, key, value):\n"
        "        if len(self._cache) >= self.maxsize:\n"
        "            self._cache.popitem(last=False)\n"
        "        self._cache[key] = value\n"
        "        self._timestamps[key] = time.monotonic()\n"
        "    def __getitem__(self, key):\n"
        "        if key not in self._cache:\n"
        "            raise KeyError(key)\n"
        "        ts = self._timestamps.get(key, 0)\n"
        "        if time.monotonic() - ts > self.ttl:\n"
        "            del self._cache[key]\n"
        "            del self._timestamps[key]\n"
        "            raise KeyError(key)\n"
        "        return self._cache[key]\n"
        "    def expire(self):\n"
        "        now = time.monotonic()\n"
        "        expired = [k for k, ts in self._timestamps.items() if now - ts > self.ttl]\n"
        "        for k in expired:\n"
        "            self._cache.pop(k, None)\n"
        "            self._timestamps.pop(k, None)\n"
    )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "cache_manager.py"), "w").write(code)
        assert grade_fix_cachetools_ttl(d) == 0.99


# ---------------------------------------------------------------------------
# implement_pybreaker
# ---------------------------------------------------------------------------

def test_pybreaker_empty():
    with tempfile.TemporaryDirectory() as d:
        assert grade_implement_pybreaker(d) == 0.01


def test_pybreaker_no_call_method():
    """Class exists but no call() → 0.20."""
    code = (
        "import time\n"
        "class CircuitBreakerError(Exception): pass\n"
        "class CircuitBreaker:\n"
        "    def __init__(self, fail_max=5, reset_timeout=60):\n"
        "        self.fail_max = fail_max\n"
        "        self.reset_timeout = reset_timeout\n"
    )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "circuit_breaker.py"), "w").write(code)
        assert grade_implement_pybreaker(d) == 0.20


def test_pybreaker_perfect():
    """Full state machine → 0.99."""
    code = (
        "import time\n"
        "class CircuitBreakerError(Exception): pass\n"
        "class CircuitBreaker:\n"
        "    def __init__(self, fail_max=5, reset_timeout=60):\n"
        "        self.fail_max = fail_max\n"
        "        self.reset_timeout = reset_timeout\n"
        "        self._fail_count = 0\n"
        "        self._state = 'closed'\n"
        "        self._opened_at = 0\n"
        "    def call(self, func, *args, **kwargs):\n"
        "        if self._state == 'open':\n"
        "            if time.monotonic() - self._opened_at >= self.reset_timeout:\n"
        "                self._state = 'half_open'\n"
        "            else:\n"
        "                raise CircuitBreakerError('Circuit is OPEN')\n"
        "        try:\n"
        "            result = func(*args, **kwargs)\n"
        "            if self._state == 'half_open':\n"
        "                self._state = 'closed'\n"
        "                self._fail_count = 0\n"
        "            return result\n"
        "        except Exception as e:\n"
        "            self._fail_count += 1\n"
        "            if self._fail_count >= self.fail_max:\n"
        "                self._state = 'open'\n"
        "                self._opened_at = time.monotonic()\n"
        "            raise\n"
    )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "circuit_breaker.py"), "w").write(code)
        assert grade_implement_pybreaker(d) == 0.99


# ---------------------------------------------------------------------------
# fix_async_race
# ---------------------------------------------------------------------------

def test_async_race_empty():
    with tempfile.TemporaryDirectory() as d:
        assert grade_fix_async_race(d) == 0.01


def test_async_race_still_racy():
    """Original buggy code → 0.25 or 0.50."""
    code = (
        "import asyncio\n"
        "class SharedCounter:\n"
        "    def __init__(self):\n"
        "        self.count = 0\n"
        "    async def increment(self):\n"
        "        current = self.count\n"
        "        await asyncio.sleep(0)\n"
        "        self.count = current + 1\n"
    )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "async_worker.py"), "w").write(code)
        score = grade_fix_async_race(d)
        assert score in (0.25, 0.50)  # race is nondeterministic


def test_async_race_perfect():
    """Lock-protected increment → 0.99."""
    code = (
        "import asyncio\n"
        "class SharedCounter:\n"
        "    def __init__(self):\n"
        "        self.count = 0\n"
        "        self._lock = asyncio.Lock()\n"
        "    async def increment(self):\n"
        "        async with self._lock:\n"
        "            self.count += 1\n"
    )
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "async_worker.py"), "w").write(code)
        assert grade_fix_async_race(d) == 0.99
