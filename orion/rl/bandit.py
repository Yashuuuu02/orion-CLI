import json
import os
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

ACTIONS = [
    {"planner": "fast", "coder": "fast", "reviewer": False},
    {"planner": "fast", "coder": "coder", "reviewer": False},
    {"planner": "fast", "coder": "coder", "reviewer": True},
    {"planner": "balanced", "coder": "coder", "reviewer": True},
    {"planner": "fast", "coder": "fast", "reviewer": True},
    {"planner": "fast", "coder": "coder", "reviewer": True},
    {"planner": "balanced", "coder": "coder", "reviewer": True},
    {"planner": "balanced", "coder": "heavy", "reviewer": False},
]

ACTION_NAMES = [
    "fast-fast-no-review",
    "fast-coder-no-review",
    "fast-coder-balanced-review",
    "balanced-coder-balanced-review",
    "fast-fast-with-review",
    "fast-coder-with-review",
    "balanced-coder-with-review",
    "balanced-heavy-no-review",
]

ALPHA = 1.0


@dataclass
class BanditWeights:
    weights: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    total_count: int = 0


class LinUCBBandit:
    def __init__(self, weights_file: Optional[str] = None, alpha: float = ALPHA):
        self.weights_file = weights_file or os.path.expanduser("~/.orion/bandit_weights.json")
        self.alpha = alpha
        self.n_actions = len(ACTIONS)
        self.n_features = 4
        self.weights = {}
        self.counts = {}
        self.total_count = 0
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, "r") as f:
                    data = json.load(f)
                    self.weights = {int(k): np.array(v) for k, v in data.get("weights", {}).items()}
                    self.counts = data.get("counts", {})
                    self.total_count = data.get("total_count", 0)
        except Exception:
            pass

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
            weights_serializable = {k: v.tolist() for k, v in self.weights.items()}
            with open(self.weights_file, "w") as f:
                json.dump({
                    "weights": weights_serializable,
                    "counts": self.counts,
                    "total_count": self.total_count,
                }, f)
        except Exception:
            pass

    def _get_weight(self, action: int) -> np.ndarray:
        if action not in self.weights:
            self.weights[action] = np.zeros(self.n_features)
            self.counts[action] = 0
        return self.weights[action]

    def _ucb_score(self, state: np.ndarray, action: int) -> float:
        weight = self._get_weight(action)
        count = self.counts.get(action, 0)

        if count == 0:
            return float("inf")

        expected = np.dot(weight, state)
        exploration = self.alpha * np.sqrt(np.log(self.total_count + 1) / (count + 1))

        return expected + exploration

    def select(self, state: list) -> int:
        state = np.array(state, dtype=float)
        self.total_count += 1

        scores = [self._ucb_score(state, a) for a in range(self.n_actions)]
        best = int(np.argmax(scores))

        if best not in self.counts:
            self.counts[best] = 0
        self.counts[best] += 1

        return best

    def get_action(self, action_idx: int) -> dict:
        return ACTIONS[action_idx]

    def get_action_name(self, action_idx: int) -> str:
        return ACTION_NAMES[action_idx]

    def update(self, state: list, action: int, reward: float) -> None:
        state = np.array(state, dtype=float)
        weight = self._get_weight(action)
        count = self.counts.get(action, 0)

        predicted = np.dot(weight, state)
        error = reward - predicted

        learning_rate = 1.0 / (count + 1)
        weight += learning_rate * error * state

        self.weights[action] = weight
        self.counts[action] = count + 1
        self._save()

    def reset(self) -> None:
        self.weights = {}
        self.counts = {}
        self.total_count = 0
        self._save()


def get_default_action() -> dict:
    return {"coder_tier": "coder"}


def action_to_pipeline(action: dict) -> dict:
    return {
        "coder_tier": action.get("coder", "coder"),
    }