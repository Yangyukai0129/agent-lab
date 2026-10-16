from agent_lab.stage1_single_agent.tools import TOOL_DEFINITIONS, run_tool

# 每個「會呼叫模型」的角色可以用哪些工具。沒列出的一律不能用（預設拒絕）。
ROLE_TOOLS = {
    "planner": {"list_dir", "read_file", "run_command"},  # 唯讀：不能寫檔
    "executor": {"list_dir", "read_file", "run_command", "write_file"},
}


def _allowed(role: str) -> set[str]:
    if role not in ROLE_TOOLS:
        raise ValueError(f"未知的角色：{role}。可用角色：{', '.join(ROLE_TOOLS)}")
    return ROLE_TOOLS[role]


def tool_definitions_for(role: str) -> list[dict]:
    """給模型看的工具清單：只包含這個角色能用的（軟防線）。"""
    allowed = _allowed(role)
    return [d for d in TOOL_DEFINITIONS if d["function"]["name"] in allowed]


def run_tool_as(role: str, name: str, arguments: dict) -> str:
    """執行工具前再檢查一次角色權限（硬防線）：模型就算亂要求也會被擋。"""
    allowed = _allowed(role)
    if name not in allowed:
        return (
            f"錯誤：角色 '{role}' 不能使用工具 '{name}'。"
            f"可用工具：{', '.join(sorted(allowed))}"
        )
    return run_tool(name, arguments)