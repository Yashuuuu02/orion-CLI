import os
import re

class ReadTool:
    def execute(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                if len(content) > 50000:
                    return content[:50000] + "\n[file truncated at 50k chars]"
                return content
        except Exception:
            return ""

class WriteTool:
    def execute(self, path: str, content: str) -> bool:
        try:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception:
            return False

class EditTool:
    def execute(self, path: str, content: str) -> bool:
        try:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception:
            return False

class GrepTool:
    IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".env",
                   "dist", "build", "orion_cli.egg-info"}

    def execute(self, pattern: str, search_path: str = ".") -> str:
        try:
            regex = re.compile(pattern)
        except Exception as e:
            return f"GrepTool error: {e}"

        matches = []
        try:
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            for idx, line in enumerate(f, 1):
                                if regex.search(line):
                                    matches.append(f"{filepath}:{idx}: {line.rstrip()}")
                                    if len(matches) == 101:
                                        break
                    except Exception:
                        pass
                    if len(matches) == 101:
                        break
                if len(matches) == 101:
                    break

            if not matches:
                return f"No matches found for pattern: {pattern}"

            if len(matches) > 100:
                return "\n".join(matches[:100]) + "\n[truncated — showing first 100 matches]"
            return "\n".join(matches)
        except Exception as e:
            return f"GrepTool error: {e}"
