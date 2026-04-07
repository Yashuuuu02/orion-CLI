import re
import os
from dataclasses import dataclass
from typing import Optional

INTENT_MAP = {
    "bug_fix": 0,
    "feature": 1,
    "refactor": 2,
    "explain": 3,
    "test": 4,
}

COMPLEXITY_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

LANGUAGE_EXT_MAP = {
    ".py": 0,
    ".js": 1,
    ".ts": 2,
    ".jsx": 2,
    ".tsx": 2,
    ".go": 3,
    ".rs": 4,
    ".java": 5,
    ".cpp": 6,
    ".c": 6,
    ".rb": 7,
    ".php": 8,
    ".swift": 9,
    ".kt": 10,
}


def detect_language(prompt: str, cwd: str = ".") -> int:
    prompt_lower = prompt.lower()
    for ext, lang_id in LANGUAGE_EXT_MAP.items():
        if ext in prompt_lower:
            return lang_id
    try:
        files = os.listdir(cwd)
        for f in files:
            if f.endswith(".py"):
                return 0
            elif f.endswith((".js", ".jsx")):
                return 1
            elif f.endswith((".ts", ".tsx")):
                return 2
            elif f.endswith(".go"):
                return 3
            elif f.endswith(".rs"):
                return 4
    except Exception:
        pass
    return 0


@dataclass
class StateVector:
    intent_type_int: int
    complexity_int: int
    language_int: int
    past_iisg_avg: float

    def to_list(self) -> list:
        return [
            self.intent_type_int,
            self.complexity_int,
            self.language_int,
            self.past_iisg_avg,
        ]


class StateEncoder:
    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or os.path.expanduser("~/.orion/iisg_history.json")
        self._history: list[float] = []

    def load_history(self) -> list[float]:
        try:
            if os.path.exists(self.history_file):
                import json
                with open(self.history_file, "r") as f:
                    self._history = json.load(f)
        except Exception:
            self._history = []
        return self._history

    def save_history(self, iisg_score: float) -> None:
        import json
        self._history.append(iisg_score)
        if len(self._history) > 5:
            self._history = self._history[-5:]
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump(self._history, f)
        except Exception:
            pass

    def encode(self, intent_type: str, complexity: str, prompt: str = "", cwd: str = ".") -> StateVector:
        intent_int = INTENT_MAP.get(intent_type, 1)
        complexity_int = COMPLEXITY_MAP.get(complexity, 1)
        language_int = detect_language(prompt, cwd)

        self.load_history()
        past_avg = sum(self._history) / len(self._history) if self._history else 0.5

        return StateVector(
            intent_type_int=intent_int,
            complexity_int=complexity_int,
            language_int=language_int,
            past_iisg_avg=past_avg,
        )