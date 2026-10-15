import os
import subprocess

from agent_lab.config import API_KEY_ENV_VAR, PROJECT_DIR
from agent_lab.stage1_single_agent.config import (
    COMMAND_TIMEOUT_SECONDS,
    MAX_TOOL_OUTPUT_CHARS,
)
from agent_lab.stage1_single_agent.permissions import (
    CommandNotAllowedError,
    PathNotAllowedError,
    resolve_safe_path,
    validate_command,
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


# 工具名稱 -> 實際函式
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "run_command": run_command,
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
                "且不能使用會修改檔案的選項（例如 --fix）。"
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
]


def run_tool(name: str, arguments: dict) -> str:
    """統一入口：執行工具並一律回傳字串，錯誤也以字串回報，讓模型能自行調整。"""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"錯誤：沒有名為 '{name}' 的工具。可用工具：{', '.join(TOOL_FUNCTIONS)}"
    try:
        return func(**arguments)
    except (PathNotAllowedError, CommandNotAllowedError) as e:
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