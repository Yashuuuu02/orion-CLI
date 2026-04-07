import os
import platform
from datetime import datetime

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".env",
               "dist", "build", "orion_cli.egg-info", ".idea", ".vscode"}

TOOL_FORMAT_INSTRUCTIONS = """
You have access to these tools. Use them by outputting EXACTLY this XML format:

To READ a file:
<tool>ReadTool</tool>
<path>relative/path/to/file.py</path>

To WRITE a new file (or overwrite):
<tool>WriteTool</tool>
<path>relative/path/to/file.py</path>
<content>
# complete file content here — not a diff, the entire file
</content>

To EDIT an existing file (full overwrite):
<tool>EditTool</tool>
<path>relative/path/to/file.py</path>
<content>
# complete new file content
</content>

To SEARCH for a pattern across files:
<tool>GrepTool</tool>
<path>search_pattern_here</path>

Rules:
- Use relative paths only. Never absolute paths like /home/ or C:\\.
- For WriteTool and EditTool: output the COMPLETE file content, not a diff.
- You may output multiple tool blocks in one response.
- Read files before editing them when you need to understand existing code.
- Do not use markdown code fences (no triple backticks).
- Output tool blocks and brief explanations only. No preamble.
"""

def build_file_tree(cwd: str = ".") -> str:
    lines = ["Project structure:"]
    try:
        entries = sorted(os.listdir(cwd))
        for entry in entries:
            if entry in IGNORE_DIRS or entry.startswith("."):
                continue
            full = os.path.join(cwd, entry)
            if os.path.isdir(full):
                lines.append(f"├── {entry}/")
                try:
                    sub_entries = sorted(os.listdir(full))[:10]
                    for sub in sub_entries:
                        if sub not in IGNORE_DIRS and not sub.startswith("."):
                            lines.append(f"│   ├── {sub}")
                except PermissionError:
                    pass
            else:
                lines.append(f"├── {entry}")
    except Exception:
        lines.append("(unable to read directory)")
    return "\n".join(lines)

def build_system_prompt(intent, cwd: str = None) -> str:
    cwd = cwd or os.getcwd()
    file_tree = build_file_tree(cwd)
    intent_type = intent.intent_type if intent else "feature"
    complexity = intent.complexity if intent else "medium"

    return f"""You are Orion, an expert AI coding assistant running in a terminal.

Environment:
- Working directory: {cwd}
- Platform: {platform.system()} {platform.release()}
- Date: {datetime.now().strftime("%Y-%m-%d")}
- Task intent: {intent_type} (complexity: {complexity})

{file_tree}

{TOOL_FORMAT_INSTRUCTIONS}"""

def build_messages(
    prompt: str,
    intent,
    conversation_history: list,
    iteration_feedback: str = None,
) -> list:
    messages = []
    messages.append({"role": "system", "content": build_system_prompt(intent)})

    history = conversation_history or []
    if len(history) > 10:
        earlier = history[:-10]
        user_count = sum(1 for m in earlier if m["role"] == "user")
        summary = (
            f"[Earlier in this session: {user_count} prior exchanges. "
            f"Last topic: {earlier[-1]['content'][:100]}...]"
        )
        messages.append({"role": "system", "content": summary})
        history = history[-10:]

    for msg in history:
        content = msg["content"]
        if len(content) > 2000:
            content = content[:2000] + "\n[truncated]"
        messages.append({"role": msg["role"], "content": content})

    messages.append({"role": "user", "content": prompt})

    if iteration_feedback:
        messages.append({
            "role": "user",
            "content": (
                f"Your previous output failed these validation checks:\n{iteration_feedback}\n"
                f"Fix these issues in your next response."
            )
        })

    return messages
