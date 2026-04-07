import asyncio
import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from orion.provider.provider import Provider
from orion.tool.tools import ReadTool, WriteTool, EditTool, GrepTool
from orion.pipeline.models import IntentResult
from orion.pipeline import agentic_loop

api_key = os.environ.get("NVIDIA_NIM_API_KEY", "").strip()
if not api_key:
    print("ERROR: NVIDIA_NIM_API_KEY not set")
    sys.exit(1)

p = Provider(api_key=api_key)
tools = {
    "read":  ReadTool(),
    "write": WriteTool(),
    "edit":  EditTool(),
    "grep":  GrepTool(),
}
intent = IntentResult("feature", "low", "")
action = {"coder_tier": "coder"}

async def main():
    result = await agentic_loop.run(
        prompt="write a python function called multiply that takes two numbers and returns their product. save it to multiply.py",
        intent=intent,
        action=action,
        provider=p,
        tools=tools,
        on_stage=print,
    )
    print("Iterations:", len(result.iterations))
    print("Tool calls:", len(result.tool_calls))
    print("Final response (first 200):", result.final_response[:200])
    r = tools["read"].execute("multiply.py")
    print("multiply.py exists:", bool(r))
    print("AGENTIC LOOP OK")

asyncio.run(main())
