from dataclasses import dataclass
from typing import Optional
import os
import random
import sys
import builtins


@dataclass
class Task:
    name: str
    difficulty: str
    prompt: str
    setup_files: dict
    grader: callable


SAFED_BUILTINS = {
    k: v for k, v in builtins.__dict__.items() 
    if k not in ('eval', 'exec', 'compile', '__import__', 'open', 'file', 'input')
}
SAFED_BUILTINS['print'] = lambda *args, **kwargs: None  # Suppress output


RESTRICTED_MODULES = {
    'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
    'http', 'ftplib', 'telnetlib', 'poplib', 'imaplib',
    'smtplib', 'xml', 'pickle', 'shelve', 'msvcrt',
    'termios', 'tty', 'pty', 'ptyprocess', 'signal',
    'multiprocessing', 'threading', 'concurrent', 'asyncio',
}


def _safe_exec(code: str, path: str) -> dict:
    namespace = {
        '__builtins__': SAFED_BUILTINS,
        '__name__': '__grader__',
        '__doc__': None,
    }
    
    restricted_builtins = {}
    for name in RESTRICTED_MODULES:
        restricted_builtins[name] = None
    
    try:
        compiled = compile(code, path, 'exec')
        exec(compiled, namespace)
        return namespace
    except Exception as e:
        return {"_error": str(e)}


def grade_easy(workspace: str) -> float:
    utils_path = os.path.join(workspace, "utils.py")
    if not os.path.exists(utils_path):
        return 0.01

    try:
        code = open(utils_path).read()
        compile(code, utils_path, "exec")
    except SyntaxError:
        return 0.01

    try:
        namespace = _safe_exec(code, utils_path)
        if "_error" in namespace:
            return 0.01
        if "add" in namespace:
            result = namespace["add"](2, 3)
            if result == 5:
                return 0.99
            return 0.5
        return 0.25
    except Exception:
        return 0.25


def grade_medium(workspace: str) -> float:
    buggy_path = os.path.join(workspace, "buggy.py")
    if not os.path.exists(buggy_path):
        return 0.01

    try:
        code = open(buggy_path).read()
        compile(code, buggy_path, "exec")
    except SyntaxError:
        return 0.01

    try:
        namespace = _safe_exec(code, buggy_path)
        if "_error" in namespace:
            return 0.01
        if "add" in namespace:
            result = namespace["add"](2, 3)
            if result is None:
                return 0.01
            if result == 5:
                return 0.99
            return 0.5
        return 0.25
    except Exception:
        return 0.25


def grade_hard(workspace: str) -> float:
    monolith_path = os.path.join(workspace, "monolith.py")
    if not os.path.exists(monolith_path):
        return 0.01

    try:
        code = open(monolith_path).read()
        compile(code, monolith_path, "exec")
    except SyntaxError:
        return 0.01

    try:
        namespace = _safe_exec(code, monolith_path)
        if "_error" in namespace:
            return 0.01

        has_parse = "parse_input" in namespace
        has_process = "process_data" in namespace
        has_format = "format_output" in namespace

        if not (has_parse and has_process and has_format):
            return 0.3

        test_input = "hello world"
        try:
            parsed = namespace["parse_input"](test_input)
            processed = namespace["process_data"](parsed)
            formatted = namespace["format_output"](processed)

            if formatted and isinstance(formatted, str):
                return 0.99
            return 0.7
        except Exception:
            return 0.5
    except Exception:
        return 0.3


TASKS = [
    Task(
        name="easy_add",
        difficulty="Easy",
        prompt="Write a Python function called add(a, b) that returns the sum of a and b. Save it to utils.py.",
        setup_files={},
        grader=grade_easy,
    ),
    Task(
        name="medium_fix",
        difficulty="Medium",
        prompt="Fix the buggy.py file. The add function currently returns None instead of the sum. Make it return a + b.",
        setup_files={
            "buggy.py": "def add(a, b):\n    return None\n",
        },
        grader=grade_medium,
    ),
    Task(
        name="hard_refactor",
        difficulty="Hard",
        prompt="Refactor the monolith.py file. Create three functions: parse_input(data), process_data(data), and format_output(data). Keep the original logic but split it into these three functions.",
        setup_files={
            "monolith.py": """def process_data(input_data):
    cleaned = input_data.strip().lower()
    words = cleaned.split()
    count = len(words)
    return {'words': words, 'count': count}

def format_output(data):
    return f"Found {data['count']} words: {', '.join(data['words'])}"

def main():
    result = process_data("  Hello World This Is A Test  ")
    print(format_output(result))

if __name__ == "__main__":
    main()
""",
        },
        grader=grade_hard,
    ),
]


class TaskBank:
    def __init__(self):
        self.tasks = TASKS
        self.current_task: Optional[Task] = None

    def sample(self, difficulty: Optional[str] = None) -> Task:
        if difficulty:
            filtered = [t for t in self.tasks if t.difficulty.lower() == difficulty.lower()]
            if filtered:
                self.current_task = random.choice(filtered)
            else:
                self.current_task = random.choice(self.tasks)
        else:
            self.current_task = random.choice(self.tasks)
        return self.current_task

    def get_current(self) -> Optional[Task]:
        return self.current_task

    def grade(self, workspace: str) -> float:
        if not self.current_task:
            return 0.0
        return self.current_task.grader(workspace)

    def reset(self) -> dict:
        self.current_task = None
        return {"status": "reset", "task": None}


def get_task_bank() -> TaskBank:
    return TaskBank()