import sys
from pathlib import Path
import re

from agent_lab.config import PROJECT_DIR


class PathNotAllowedError(Exception):
    """路徑跑出 sample_project 之外時丟出。"""

class ContentNotAllowedError(Exception):
    """寫入的內容含有不允許的東西（例如用註解關掉檢查）時丟出。"""

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

# ---- 寫檔政策與確認 ----

# 這些檔案是「裁判」：模型不能改，否則可以靠放寬規則或改測試來「通過」
_PROTECTED_NAMES = {"conftest.py", "ruff.toml", "pytest.ini"}
_PROTECTED_DIRS = {"tests"}


def check_write_allowed(relative_path: str) -> Path:
    """寫檔前的硬性檢查，通過才回傳安全路徑；不通過丟 PathNotAllowedError。"""
    target = resolve_safe_path(relative_path)
    rel = target.relative_to(PROJECT_DIR)
    if target.name in _PROTECTED_NAMES or _PROTECTED_DIRS & set(rel.parts):
        raise PathNotAllowedError(
            f"'{relative_path}' 是受保護的檔案（測試或設定檔），不允許修改。"
        )
    if target.suffix != ".py":
        raise PathNotAllowedError("只允許修改 .py 檔案。")
    if not target.is_file():
        raise PathNotAllowedError(
            f"'{relative_path}' 不存在。目前只允許修改既有檔案，不能新增檔案。"
        )
    return target


def confirm_write(relative_path: str, diff_text: str) -> bool:
    """顯示 diff 並詢問 y/n。只有明確輸入 y 才算同意；無法互動時一律拒絕。"""
    print(f"\n--- 模型想修改 {relative_path} ---")
    print(diff_text)
    try:
        answer = input(f"允許寫入 {relative_path}？[y/N] ")
    except EOFError:
        print("(無法互動，預設拒絕)")
        return False
    return answer.strip().lower() == "y"

# 用註解「關掉檢查」的寫法：這會讓 ruff 變綠，卻沒有真的修好問題
_SUPPRESSION = re.compile(
    r"#\s*(noqa|ruff:\s*(noqa|disable)|type:\s*ignore|fmt:\s*(off|skip)|pylint:\s*disable)",
    re.IGNORECASE,
)


def check_no_new_suppression(old_text: str, new_text: str) -> None:
    """新內容的抑制註解數量比舊內容多，就拒絕。"""
    if len(_SUPPRESSION.findall(new_text)) > len(_SUPPRESSION.findall(old_text)):
        raise ContentNotAllowedError(
            "不允許新增 noqa、type: ignore 等抑制檢查的註解。"
            "請真正修正問題（例如刪除未使用的 import），而不是讓檢查器略過它。"
        )