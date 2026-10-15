from agent_lab.stage1_single_agent.tools import run_tool, _truncate

print(run_tool("list_dir", {}))
print(run_tool("read_file", {"path": "calculator.py"}))
print(run_tool("read_file", {"path": "../config.py"}))
print(run_tool("read_file", {"path": r"C:\Windows"}))
print(run_tool("read_file", {"path": "nope.py"}))
print(run_tool("read_file", {"path": "tests"}))
print(run_tool("delete_file", {"path": "x"}))
print(_truncate("x" * 5000)[-60:])