from textual.widgets import DirectoryTree
from pathlib import Path

IGNORE = {".git", "node_modules", "__pycache__", ".venv", ".env", "dist", "build"}


class OrionFileTree(DirectoryTree):
    def filter_paths(self, paths):
        return [
            p for p in paths
            if p.name not in IGNORE
            and not p.name.endswith(".pyc")
            and p.name != ".DS_Store"
        ]
