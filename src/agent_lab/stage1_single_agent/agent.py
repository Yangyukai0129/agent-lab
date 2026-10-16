import json
import sys

import requests

from agent_lab.config import MODEL, OPENROUTER_BASE_URL, get_api_key
from agent_lab.stage1_single_agent.config import MAX_TURNS
from agent_lab.stage1_single_agent.tools import TOOL_DEFINITIONS, run_tool
from agent_lab.stage1_single_agent.logger import RunLogger

REQUEST_TIMEOUT_SECONDS = 120
PREVIEW_CHARS = 200

SYSTEM_PROMPT = """你是一個協助修正 Python 專案 lint 問題的程式碼助理。
你只能透過提供的工具存取專案目錄內的檔案，路徑一律使用相對於專案根目錄的相對路徑。
可用工具：list_dir、read_file、run_command（只能執行 ruff check 與 pytest）、write_file。
write_file 需要提供「修改後的完整檔案內容」，不要省略沒改的部分；每次寫入使用者都會審核，可能被拒絕。
不可以修改 tests/、ruff.toml、pytest.ini；要讓檢查通過，只能修正原始碼本身。
不可以用 noqa、type: ignore 等註解讓檢查通過，要真正修正問題（例如刪除未使用的 import）。
第一次執行 ruff check 後，先把所有問題整理成清單，再對同一個檔案一次修完，不要一個一個分次寫入。
如果使用者拒絕你的修改，不要改用其他繞過方式；請說明你想做什麼，或改成範圍較小的修改。
執行 ruff check 時請加上 --output-format=concise。
每次修改後都要重新執行 ruff check 與 pytest，兩者都通過才算完成。
不要假設檔案內容或執行結果，需要時先用工具確認。
每次呼叫工具前，先用一句話說明這一步的目的。
回答請使用繁體中文，並簡潔說明你的發現。"""


class ModelCallError(Exception):
    """呼叫模型失敗（連線、認證、額度、模型生成途中出錯等）。"""


def _error_message(obj) -> str:
    """從回應（或 choice）裡取出 error.message。"""
    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err) if err else "（沒有錯誤訊息）"


def call_model(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """送出一次請求，回傳解析後的回應；任何失敗都轉成 ModelCallError。"""
    if not MODEL:
        raise ModelCallError(
            "config.py 的 MODEL 還是空的，請填入支援 tools 的模型名稱。"
        )

    api_key = get_api_key()  # 缺金鑰時會丟 RuntimeError，由 main() 處理
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOL_DEFINITIONS if tools is None else tools,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise ModelCallError(f"連線失敗：{e}") from e

    try:
        data = resp.json()
    except ValueError:
        raise ModelCallError(f"HTTP {resp.status_code}：回應不是合法的 JSON") from None
    if not isinstance(data, dict):
        raise ModelCallError(f"HTTP {resp.status_code}：回應格式不符預期")

    # 錯誤可能是非 200 狀態，也可能藏在 200 的內容裡，兩種都要檢查
    if resp.status_code != 200:
        raise ModelCallError(f"HTTP {resp.status_code}：{_error_message(data)}")
    if not data.get("choices"):
        raise ModelCallError(f"回應沒有 choices：{_error_message(data)}")
    choice = data["choices"][0]
    if choice.get("error") or choice.get("finish_reason") == "error":
        raise ModelCallError(f"模型生成途中出錯：{_error_message(choice)}")
    return data


def _preview(text, limit: int = PREVIEW_CHARS) -> str:
    """把多行文字壓成一行並截短，只用於終端機顯示。"""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _execute_tool_call(call: dict, logger: RunLogger, turn: int) -> str:
    """解析一個工具請求並執行；任何問題都回傳錯誤字串給模型，不丟例外。"""
    func = call.get("function") or {}
    name = func.get("name", "")
    try:
        arguments = json.loads(func.get("arguments") or "{}")
    except json.JSONDecodeError:
        print(f"  模型要求：{name}（參數不是合法 JSON）")
        result = "錯誤：工具參數不是合法的 JSON，請重新呼叫並提供正確格式。"
        logger.log("tool_result", turn=turn, name=name,
                   arguments=func.get("arguments"), result=result)
        return result
    if not isinstance(arguments, dict):
        result = "錯誤：工具參數必須是 JSON 物件。"
        logger.log("tool_result", turn=turn, name=name,
                   arguments=func.get("arguments"), result=result)
        return result

    print(f"  模型要求：{name}({_preview(json.dumps(arguments, ensure_ascii=False))})")
    result = run_tool(name, arguments)
    print(f"  結果：{_preview(result)}")
    logger.log("tool_result", turn=turn, name=name, arguments=arguments, result=result)
    return result


def _print_usage(turn: int, response: dict, message_count: int) -> None:
    usage = response.get("usage") or {}
    print(
        f"[第 {turn} 輪] 用量：prompt={usage.get('prompt_tokens', '?')} "
        f"completion={usage.get('completion_tokens', '?')} "
        f"total={usage.get('total_tokens', '?')}｜messages={message_count}"
    )


def run_agent(task: str) -> str:
    logger = RunLogger()
    if logger.path:
        print(f"日誌：{logger.path}")
    logger.log("run_start", task=task, model=MODEL, max_turns=MAX_TURNS)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    totals = {"prompt": 0, "completion": 0, "total": 0}
    reason, error, turns = "error", None, 0

    try:
        for turn in range(1, MAX_TURNS + 1):
            turns = turn
            response = call_model(messages)
            message = response["choices"][0]["message"]
            messages.append(message)
            _print_usage(turn, response, len(messages))

            usage = response.get("usage") or {}
            used = {
                "prompt": usage.get("prompt_tokens") or 0,
                "completion": usage.get("completion_tokens") or 0,
                "total": usage.get("total_tokens") or 0,
            }
            for k in totals:
                totals[k] += used[k]
            logger.log(
                "model_response",
                turn=turn,
                usage=used,
                message_count=len(messages),
                content=message.get("content"),
                tool_calls=message.get("tool_calls"),
            )

            if message.get("content"):
                print(f"  模型說：{_preview(message['content'])}")

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                reason = "finished"
                return message.get("content") or ""

            for call in tool_calls:
                result = _execute_tool_call(call, logger, turn)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": call.get("function", {}).get("name", ""),
                        "content": result,
                    }
                )

        reason = "max_turns"
        return f"已達輪數上限（{MAX_TURNS} 輪），停止。"
    except BaseException as e:  # 包含 Ctrl+C，讓中斷也留下紀錄
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        logger.log("run_end", reason=reason, turns=turns, tokens=totals, error=error)


def main() -> None:
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = input("請輸入任務：").strip()
    if not task:
        print("沒有輸入任務，結束。")
        return

    try:
        answer = run_agent(task)
    except (ModelCallError, RuntimeError) as e:
        print(f"執行失敗：{e}")
        sys.exit(1)

    print("\n=== 最終回答 ===")
    print(answer)


if __name__ == "__main__":
    main()