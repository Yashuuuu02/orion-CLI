import re
from orion.pipeline.models import (
    AgentResult, LoopIteration, ToolCall, ToolResult
)
from orion.pipeline import c09_validation
from orion.pipeline.context import build_messages

MAX_ITERATIONS = 3

TOOL_BLOCK_RE = re.compile(
    r'<tool>(.*?)</tool>\s*<path>(.*?)</path>(?:\s*<content>(.*?)</content>)?',
    re.DOTALL
)

def parse_tool_calls(text: str) -> list:
    calls = []
    for m in TOOL_BLOCK_RE.finditer(text):
        tool    = m.group(1).strip()
        path    = m.group(2).strip()
        content = (m.group(3) or "").strip()
        if tool in ("ReadTool", "WriteTool", "EditTool", "GrepTool"):
            calls.append(ToolCall(tool=tool, path=path, content=content))
    return calls

def execute_tool_calls(tool_calls: list, tools: dict) -> list:
    results = []
    for tc in tool_calls:
        if tc.tool == "ReadTool":
            output = tools["read"].execute(tc.path)
            results.append(ToolResult(tc, success=bool(output), output=output))
        elif tc.tool == "WriteTool":
            ok = tools["write"].execute(tc.path, tc.content)
            results.append(ToolResult(tc, success=ok, output=""))
        elif tc.tool == "EditTool":
            ok = tools["edit"].execute(tc.path, tc.content)
            results.append(ToolResult(tc, success=ok, output=""))
        elif tc.tool == "GrepTool":
            output = tools["grep"].execute(tc.path)
            results.append(ToolResult(tc, success=True, output=output))
    return results

async def run(
    prompt: str,
    intent,
    action: dict,
    provider,
    tools: dict,
    on_stage=lambda s: None,
    on_token_delta=None,
    conversation_history: list = None,
) -> AgentResult:
    conversation_history = conversation_history or []
    all_tool_calls = []
    iterations = []
    last_validation = None

    for i in range(MAX_ITERATIONS):
        attempt_num = i + 1
        on_stage(f"💻 Coding... (attempt {attempt_num}/{MAX_ITERATIONS})")

        # Build iteration feedback from C09 failures
        iteration_feedback = None
        if last_validation and last_validation.pass_rate < 1.0:
            failing = [
                k for k, v in last_validation.clause_results.items() if not v
            ]
            iteration_feedback = "\n".join(
                f"- {clause}: FAILED" for clause in failing
            )

        messages = build_messages(
            prompt=prompt,
            intent=intent,
            conversation_history=conversation_history,
            iteration_feedback=iteration_feedback,
        )

        # Stream only on first iteration
        should_stream = (i == 0)
        response = await provider.complete(
            action.get("coder_tier", "coder"),
            messages,
            stream=should_stream,
            on_token_delta=on_token_delta if should_stream else None,
        )

        # Parse tool calls
        tool_calls = parse_tool_calls(response)

        # Auto-retry if response is long but no tool calls parsed
        if not tool_calls and len(response.strip()) > 300:
            on_stage("⚠️ Fixing tool format...")
            retry_messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content":
                 "You did not use the required tool format. "
                 "Rewrite your response using <tool>WriteTool</tool> etc. "
                 "Do not use markdown code fences."},
            ]
            response = await provider.complete(
                action.get("coder_tier", "coder"),
                retry_messages,
                stream=False,
            )
            tool_calls = parse_tool_calls(response)

        # Execute tools
        results = execute_tool_calls(tool_calls, tools)
        all_tool_calls.extend(tool_calls)

        # Feed read results back into next iteration context
        read_results = [r for r in results if r.tool_call.tool == "ReadTool" and r.output]
        if read_results and i < MAX_ITERATIONS - 1:
            for rr in read_results:
                conversation_history = conversation_history + [
                    {"role": "system",
                     "content": f"[ReadTool result for {rr.tool_call.path}]:\n{rr.output[:1000]}"}
                ]

        # Run C09 on this iteration's writes
        write_calls = [tc for tc in tool_calls
                       if tc.tool in ("WriteTool", "EditTool")]
        last_validation = c09_validation.run(write_calls)

        iterations.append(LoopIteration(
            iteration=attempt_num,
            response=response,
            tool_calls=tool_calls,
            tool_results=results,
            validation=last_validation,
        ))

        on_stage(
            f"✅ Validation: {last_validation.pass_rate:.0%} IISG "
            f"({'pass' if last_validation.syntax_valid else 'syntax error'})"
        )

        if last_validation.pass_rate == 1.0:
            break

    tokens_used = sum(len(it.response.split()) for it in iterations) * 2

    return AgentResult(
        iterations=iterations,
        final_response=iterations[-1].response if iterations else "",
        tool_calls=all_tool_calls,
        tokens_used=tokens_used,
    )
