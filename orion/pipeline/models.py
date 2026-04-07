from dataclasses import dataclass, field

@dataclass
class IntentResult:
    intent_type: str   # bug_fix | feature | refactor | explain | test
    complexity: str    # low | medium | high
    raw: str           # raw LLM response

@dataclass
class ToolCall:
    tool: str          # ReadTool | WriteTool | EditTool | GrepTool
    path: str          # file path or grep pattern
    content: str       # content for Write/Edit; empty for Read/Grep

@dataclass
class ToolResult:
    tool_call: ToolCall
    success: bool
    output: str        # read content, grep results, or "" for writes

@dataclass
class LoopIteration:
    iteration: int
    response: str
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    validation: object = None

@dataclass
class AgentResult:
    iterations: list = field(default_factory=list)
    final_response: str = ""
    tool_calls: list = field(default_factory=list)
    tokens_used: int = 0

@dataclass
class ValidationResult:
    pass_rate: float
    syntax_valid: bool
    clause_results: dict = field(default_factory=dict)

@dataclass
class PipelineContext:
    prompt: str
    intent: object = None
    agent: object = None
    validation: object = None
    iisg_pass_rate: float = 0.0
    syntax_valid: bool = False
    tokens_used: int = 0
    reward: float = 0.0
    action: dict = field(default_factory=dict)
    final_response: str = ""
    time_taken: float = 0.0
    error: str = ""
