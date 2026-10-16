from pathlib import Path

from agent_lab.config import PROJECT_DIR
from agent_lab.stage1_single_agent.permissions import (
    PathNotAllowedError,
    resolve_safe_path,
)
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


def _check_write_scope(arguments: dict, write_scope: set[Path]) -> str | None:
    """write_file 的目標必須在寫入範圍內；不在就回傳錯誤字串，在就回傳 None。"""
    path = arguments.get("path")
    if not isinstance(path, str):
        return "錯誤：write_file 需要字串參數 path。"
    try:
        target = resolve_safe_path(path)
    except PathNotAllowedError as e:
        return f"錯誤：{e}"
    if target not in write_scope:
        names = ", ".join(sorted(p.relative_to(PROJECT_DIR).as_posix() for p in write_scope))
        return f"錯誤：'{path}' 不在這次的修改範圍內。目前只能修改：{names}"
    return None


def run_tool_as(
    role: str,
    name: str,
    arguments: dict,
    write_scope: set[Path] | None = None,
) -> str:
    """執行工具前再檢查一次角色權限與寫入範圍（硬防線）：模型亂要求也會被擋。"""
    allowed = _allowed(role)
    if name not in allowed:
        return (
            f"錯誤：角色 '{role}' 不能使用工具 '{name}'。"
            f"可用工具：{', '.join(sorted(allowed))}"
        )
    if name == "write_file" and write_scope is not None:
        error = _check_write_scope(arguments, write_scope)
        if error:
            return error
    return run_tool(name, arguments)