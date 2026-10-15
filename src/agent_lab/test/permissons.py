from agent_lab.stage1_single_agent.permissions import resolve_safe_path, PathNotAllowedError

for p in ["calculator.py", "tests/test_calculator.py", "../config.py", r"C:\Windows"]:
    try:
        print("通過", resolve_safe_path(p))
    except PathNotAllowedError as e:
        print("被擋", e)