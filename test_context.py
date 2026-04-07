from orion.pipeline.context import build_file_tree, build_system_prompt, build_messages
from orion.pipeline.models import IntentResult

intent = IntentResult('feature', 'low', 'raw')

tree = build_file_tree('.')
assert 'Project structure:' in tree
print('File tree (first 200 chars):', tree[:200])

system = build_system_prompt(intent)
assert 'Orion' in system
assert 'WriteTool' in system
assert 'ReadTool' in system
print('System prompt length:', len(system))

msgs = build_messages('write a hello world function', intent, [], None)
assert msgs[0]['role'] == 'system'
assert msgs[-1]['role'] == 'user'
assert msgs[-1]['content'] == 'write a hello world function'
print('Messages:', len(msgs), 'items')

history = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]
msgs = build_messages('fix the bug', intent, history, 'no_hardcoded_secrets failed')
assert any('validation' in m['content'].lower() for m in msgs)
print('CONTEXT OK')
