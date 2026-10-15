import os
import subprocess
import difflib

from agent_lab.config import API_KEY_ENV_VAR, PROJECT_DIR
from agent_lab.stage1_single_agent.config import (
    COMMAND_TIMEOUT_SECONDS,
    MAX_TOOL_OUTPUT_CHARS,
)
from agent_lab.stage1_single_agent.permissions import (
    CommandNotAllowedError,
    ContentNotAllowedError,
    PathNotAllowedError,
    resolve_safe_path,
    validate_command,
    check_write_allowed,
    confirm_write,
    check_no_new_suppression,
)

# list_dir 不列出這些自動產生的資料夾，避免干擾模型
_HIDDEN_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _truncate(text: str) -> str:
    """輸出太長就截斷，並明確告訴模型內容被截斷了。"""
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_TOOL_OUTPUT_CHARS]
        + f"\n[輸出已截斷：原長度 {len(text)} 字元，"
        + f"只顯示前 {MAX_TOOL_OUTPUT_CHARS} 字元]"
    )


def read_file(path: str) -> str:
    target = resolve_safe_path(path)
    return _truncate(target.read_text(encoding="utf-8"))


def list_dir(path: str = ".") -> str:
    target = resolve_safe_path(path)
    entries = [e for e in target.iterdir() if e.name not in _HIDDEN_NAMES]
    # 資料夾排前面，其餘依名稱排序
    entries.sort(key=lambda e: (e.is_file(), e.name))
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
    return _truncate("\n".join(lines) if lines else "(空目錄)")


def run_command(command: list) -> str:
    argv = validate_command(command)  # 不在白名單內會丟 CommandNotAllowedError

    # 子行程不需要 API 金鑰，所以不傳給它
    env = {k: v for k, v in os.environ.items() if k != API_KEY_ENV_VAR}
    env["PYTHONUTF8"] = "1"

    try:
        proc = subprocess.run(
            argv,  # 參數列，不經過 shell
            cwd=PROJECT_DIR,
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"錯誤：指令執行超過 {COMMAND_TIMEOUT_SECONDS} 秒，已中止。"

    output = (proc.stdout + proc.stderr).strip() or "(沒有輸出)"
    return _truncate(f"[exit code {proc.returncode}]\n{output}")


def write_file(path: str, content: str) -> str:
    """用 content 完整取代既有 .py 檔案；先顯示 diff，經使用者輸入 y 才寫入。"""
    target = check_write_allowed(path)  # 不合規直接丟錯，不會打擾使用者

    with open(target, encoding="utf-8", newline="") as f:
        old_raw = f.read()
    newline = "\r\n" if "\r\n" in old_raw else "\n"  # 保留原本換行，diff 才不會整份都變
    old_lines = old_raw.replace("\r\n", "\n").split("\n")
    new_lines = content.replace("\r\n", "\n").split("\n")

    if old_raw.endswith("\n") and new_lines[-1] != "":
        new_lines.append("")  # 原檔以換行結尾，就保證新內容也是

    check_no_new_suppression(old_raw, content)

    if old_lines == new_lines:
        return "內容沒有變動，未寫入。"

    diff = list(
        difflib.unified_diff(
            old_lines, new_lines, f"a/{path}", f"b/{path}", lineterm=""
        )
    )
    added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
    removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))

    if not confirm_write(path, "\n".join(diff)):
        return (
            "使用者拒絕了這次修改（原因未知），檔案沒有變動。"
            "不要用 noqa 等方式繞過檢查；請改成範圍較小的修改，或停下來向使用者說明。"
        )

    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write(newline.join(new_lines))
    return f"已寫入 {path}（+{added}/-{removed} 行）。請重新執行 ruff check 與 pytest 驗證。"


# 工具名稱 -> 實際函式
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "run_command": run_command,
    "write_file": write_file,
}

# 給模型看的工具描述（OpenAI 相容格式，OpenRouter 使用此格式）
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "讀取專案內的文字檔內容。路徑是相對於專案根目錄的相對路徑。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相對於專案根目錄的檔案路徑，例如 calculator.py",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出專案內某個目錄的內容，資料夾名稱結尾會加上 /。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相對於專案根目錄的目錄路徑，省略時為專案根目錄",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "在專案目錄內執行白名單內的指令，回傳 exit code 與輸出，"
                "exit code 為 0 代表成功。目前只允許 ruff check 與 pytest，"
                "且不能使用會修改檔案的選項（例如 --fix）；要改檔請用 write_file。"
                "指令以字串陣列表示，不會經過 shell。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            '指令與參數，每個元素一個字串，例如 ["ruff", "check", "."] '
                            '或 ["pytest", "-q"]。路徑請用相對於專案根目錄的相對路徑。'
                        ),
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "用你提供的完整內容取代專案內一個既有的 .py 檔案。"
                "content 必須是修改後的「整個檔案」，不能省略沒改的部分。"
                "每次寫入都會先給使用者看 diff，使用者可能拒絕。"
                "不能修改 tests/、conftest.py、ruff.toml、pytest.ini，也不能新增檔案。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相對於專案根目錄的 .py 檔案路徑"},
                    "content": {"type": "string", "description": "修改後的完整檔案內容"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def run_tool(name: str, arguments: dict) -> str:
    """統一入口：執行工具並一律回傳字串，錯誤也以字串回報，讓模型能自行調整。"""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"錯誤：沒有名為 '{name}' 的工具。可用工具：{', '.join(TOOL_FUNCTIONS)}"
    try:
        return func(**arguments)
    except (PathNotAllowedError, CommandNotAllowedError, ContentNotAllowedError) as e:
        return f"錯誤：{e}"
    except FileNotFoundError:
        return f"錯誤：找不到 '{arguments.get('path')}'。"
    except UnicodeDecodeError:
        return "錯誤：這個檔案不是 UTF-8 文字檔，無法讀取。"
    except OSError as e:
        target = arguments.get("path")
        if target:
            return f"錯誤：無法存取 '{target}'（可能是資料夾，或沒有權限）。"
        return f"錯誤：作業系統層級的失敗（{type(e).__name__}）。"
    except TypeError as e:
        return f"錯誤：工具參數不正確：{e}"
    