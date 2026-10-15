"""確認機制、指令白名單"""
from pathlib import Path

from agent_lab.config import PROJECT_DIR


class PathNotAllowedError(Exception):
    """路徑跑出 sample_project 之外時丟出。"""


def resolve_safe_path(relative_path: str) -> Path:
    resolved = (PROJECT_DIR / relative_path).resolve()
    if not resolved.is_relative_to(PROJECT_DIR):
        raise PathNotAllowedError(
            f"路徑 '{relative_path}' 不被允許：只能存取專案目錄內的檔案。"
        )
    return resolved