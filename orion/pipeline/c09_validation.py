import re
from orion.pipeline.models import ToolCall, ValidationResult

CLAUSES = {
    "no_hardcoded_secrets": re.compile(
        r'(api_key|password|token|secret|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']',
        re.IGNORECASE
    ),
    "no_dangerous_patterns": re.compile(
        r'(os\.system\s*\(|subprocess\.[a-z_]+\([^)]*shell\s*=\s*True|'
        r'\beval\s*\(|\bexec\s*\(|__import__\s*\()',
        re.IGNORECASE
    ),
    "no_todos_left": re.compile(
        r'\b(TODO|FIXME|HACK|XXX)\b',
        re.IGNORECASE
    ),
    "no_absolute_paths": re.compile(
        r'["\'](?:/home/|/root/|/Users/|/etc/|C:\\\\|C:/)',
        re.IGNORECASE
    ),
}

def _check_syntax(content: str, path: str) -> bool:
    if not path.endswith(".py"):
        return True
    try:
        compile(content, path, "exec")
        return True
    except SyntaxError:
        return False

def run(tool_calls: list) -> ValidationResult:
    write_calls = [
        tc for tc in tool_calls
        if isinstance(tc, ToolCall)
        and tc.tool in ("WriteTool", "EditTool")
        and tc.content.strip()
    ]

    if not write_calls:
        return ValidationResult(pass_rate=1.0, syntax_valid=True, clause_results={})

    clause_results = {name: True for name in CLAUSES}
    clause_results["syntax_valid"] = True
    syntax_valid = True

    for tc in write_calls:
        for name, pattern in CLAUSES.items():
            if pattern.search(tc.content):
                clause_results[name] = False
        if not _check_syntax(tc.content, tc.path):
            clause_results["syntax_valid"] = False
            syntax_valid = False

    passed = sum(1 for v in clause_results.values() if v)
    total = len(clause_results)

    return ValidationResult(
        pass_rate=round(passed / total, 4),
        syntax_valid=syntax_valid,
        clause_results=clause_results,
    )
