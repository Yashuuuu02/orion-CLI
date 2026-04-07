from orion.pipeline.c09_validation import run
from orion.pipeline.models import ToolCall

safe = ToolCall('WriteTool', 'math.py', 'def add(a, b):\n    return a + b\n')
r = run([safe])
print('Safe:', r)
assert r.pass_rate == 1.0
assert r.syntax_valid == True

bad = ToolCall('WriteTool', 'bad.py',
    'password=\"secret123\"\nos.system(\"rm -rf /\")\n# TODO fix this')
r = run([bad])
print('Bad:', r)
assert r.pass_rate < 1.0

broken = ToolCall('WriteTool', 'broken.py', 'def foo(\n    broken syntax here')
r = run([broken])
print('Broken:', r)
assert r.syntax_valid == False

assert run([]).pass_rate == 1.0
print('C09 OK')
