import json
from collections.abc import Callable
from pathlib import Path

from agent_lab.stage1_single_agent.agent import call_model
from agent_lab.stage1_single_agent.logger import RunLogger
from agent_lab.stage2_multi_agent.roles import run_tool_as, tool_definitions_for

PREVIEW_CHARS = 200

# 每次工具執行完會呼叫：(工具名稱, 參數, 結果) -> 回傳 True 代表「請立刻停止這個迴圈」
ToolHook = Callable[[str, dict, str], bool]


def _preview(text, limit: int = PREVIEW_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _execute_tool_call(
    role: str,
    call: dict,
    logger: RunLogger,
    turn: int,
    write_scope: set[Path] | None,
) -> tuple[str, str, dict | None]:
    """解析並執行一個工具請求；回傳 (結果, 工具名稱, 參數)。

    任何問題都以錯誤字串回給模型，不丟例外；參數無法解析時，第三個值是 None。
    """
    func = call.get("function") or {}
    name = func.get("name", "")
    raw_args = func.get("arguments")
    try:
        arguments = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        result = "錯誤：工具參數不是合法的 JSON，請重新呼叫並提供正確格式。"
        logger.log("tool_result", node=role, turn=turn, name=name,
                   arguments=raw_args, result=result)
        return result, name, None
    if not isinstance(arguments, dict):
        result = "錯誤：工具參數必須是 JSON 物件。"
        logger.log("tool_result", node=role, turn=turn, name=name,
                   arguments=raw_args, result=result)
        return result, name, None

    print(f"  [{role}] 要求：{name}({_preview(json.dumps(arguments, ensure_ascii=False))})")
    result = run_tool_as(role, name, arguments, write_scope)
    print(f"  [{role}] 結果：{_preview(result)}")
    logger.log("tool_result", node=role, turn=turn, name=name,
               arguments=arguments, result=result)
    return result, name, arguments


def run_role_loop(
    role: str,
    system_prompt: str,
    user_message: str,
    max_turns: int,
    logger: RunLogger,
    write_scope: set[Path] | None = None,
    on_tool_result: ToolHook | None = None,
) -> dict:
    """讓某個角色用它被允許的工具跑一個工具迴圈。

    write_scope：write_file 只能寫這些檔案（None 代表不額外限制）。
    on_tool_result：每次工具執行後呼叫，回傳 True 會立刻結束迴圈（reason 為 "stopped"）。
    回傳 {"text", "reason", "turns", "tokens"}，reason 是 finished／max_turns／stopped。
    連線、認證等系統錯誤（ModelCallError）不在這裡處理，直接往上丟。
    """
    tools = tool_definitions_for(role)  # 角色不存在會在這裡丟 ValueError
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    totals = {"prompt": 0, "completion": 0, "total": 0}

    for turn in range(1, max_turns + 1):
        response = call_model(messages, tools)
        message = response["choices"][0]["message"]
        messages.append(message)

        usage = response.get("usage") or {}
        used = {
            "prompt": usage.get("prompt_tokens") or 0,
            "completion": usage.get("completion_tokens") or 0,
            "total": usage.get("total_tokens") or 0,
        }
        for k in totals:
            totals[k] += used[k]
        print(f"[{role} 第 {turn} 輪] prompt={used['prompt']} "
              f"completion={used['completion']}｜messages={len(messages)}")
        logger.log(
            "model_response",
            node=role,
            turn=turn,
            usage=used,
            message_count=len(messages),
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
        )

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return {"text": message.get("content") or "", "reason": "finished",
                    "turns": turn, "tokens": totals}

        for call in tool_calls:
            result, name, arguments = _execute_tool_call(role, call, logger, turn, write_scope)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": call.get("function", {}).get("name", ""),
                    "content": result,
                }
            )
            if on_tool_result and arguments is not None and on_tool_result(name, arguments, result):
                return {"text": "", "reason": "stopped", "turns": turn, "tokens": totals}

    return {"text": "", "reason": "max_turns", "turns": max_turns, "tokens": totals}