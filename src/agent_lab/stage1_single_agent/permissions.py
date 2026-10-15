import sys
from pathlib import Path

from agent_lab.config import PROJECT_DIR


class PathNotAllowedError(Exception):
    """路徑跑出 sample_project 之外時丟出。"""


class CommandNotAllowedError(Exception):
    """指令或參數不在白名單內時丟出。"""


def resolve_safe_path(relative_path: str) -> Path:
    resolved = (PROJECT_DIR / relative_path).resolve()
    if not resolved.is_relative_to(PROJECT_DIR):
        raise PathNotAllowedError(
            f"路徑 '{relative_path}' 不被允許：只能存取專案目錄內的檔案。"
        )
    return resolved


# 白名單：指令名稱 -> 實際執行方式、允許的子命令、允許的旗標。
# 預設全部不行，只開放這裡明確列出的。
# 刻意不開放 --fix、format 這類會改檔的選項：改檔只能透過需要確認的 write_file。
ALLOWED_COMMANDS = {
    "ruff": {
        "base": [sys.executable, "-m", "ruff"],
        "subcommand": "check",
        "flags": {
            "--statistics",
            "--output-format=concise",
            "--output-format=full",
        },
    },
    "pytest": {
        "base": [sys.executable, "-m", "pytest"],
        "subcommand": None,
        "flags": {"-q", "-v", "-x", "--tb=short", "--tb=line", "--tb=no"},
    },
}


def _check_path_arg(arg: str) -> None:
    """位置參數必須是專案內、而且真的存在的路徑。"""
    if not arg:
        raise CommandNotAllowedError("參數不能是空字串。")
    path_part = arg.split("::", 1)[0]  # pytest 的 tests/x.py::test_name 寫法
    try:
        target = resolve_safe_path(path_part)
    except PathNotAllowedError as e:
        raise CommandNotAllowedError(str(e)) from e
    if not target.exists():
        raise CommandNotAllowedError(f"路徑 '{path_part}' 不存在。")


def validate_command(command) -> list[str]:
    """檢查指令是否在白名單內，回傳可直接交給 subprocess 的完整參數列。"""
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(a, str) for a in command)
    ):
        raise CommandNotAllowedError(
            'command 必須是非空的字串陣列，例如 ["ruff", "check", "."]。'
        )

    name, *args = command
    rule = ALLOWED_COMMANDS.get(name)
    if rule is None:
        raise CommandNotAllowedError(
            f"指令 '{name}' 不在白名單內。允許的指令：{', '.join(ALLOWED_COMMANDS)}。"
        )

    sub = rule["subcommand"]
    if sub:
        if not args or args[0] != sub:
            raise CommandNotAllowedError(f"{name} 只允許子命令 '{sub}'。")
        args = args[1:]

    for arg in args:
        if arg.startswith("-"):
            if arg not in rule["flags"]:
                allowed = ", ".join(sorted(rule["flags"]))
                raise CommandNotAllowedError(
                    f"參數 '{arg}' 不被允許。{name} 可用的旗標：{allowed}。"
                )
        else:
            _check_path_arg(arg)

    return rule["base"] + ([sub] if sub else []) + args