from orion.pipeline.models import IntentResult
from orion.provider.provider import Provider

VALID_INTENTS = {"bug_fix", "feature", "refactor", "explain", "test"}
VALID_COMPLEXITIES = {"low", "medium", "high"}

SYSTEM_PROMPT = """You are an intent classifier for a coding assistant.
Classify the user request. Respond with EXACTLY two lines and nothing else:
INTENT: <intent_type>
COMPLEXITY: <complexity>

intent_type must be one of: bug_fix, feature, refactor, explain, test
complexity must be one of: low, medium, high

No explanation. No preamble. No extra text. Exactly two lines."""

async def run(
    prompt: str,
    provider: Provider,
    on_stage = lambda s: None,
) -> IntentResult:
    on_stage("🔍 Analyzing intent...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    raw = await provider.complete("fast", messages, stream=False)

    # Parse — never crash, always return valid defaults
    intent_type = "feature"
    complexity  = "medium"
    for line in raw.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip().lower()
        if key == "INTENT" and val in VALID_INTENTS:
            intent_type = val
        elif key == "COMPLEXITY" and val in VALID_COMPLEXITIES:
            complexity = val

    on_stage(f"✅ Intent: {intent_type} ({complexity})")
    return IntentResult(intent_type=intent_type, complexity=complexity, raw=raw)
