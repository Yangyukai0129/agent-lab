import json
import re
import sys

from agent_lab.stage1_single_agent.agent import ModelCallError
from agent_lab.stage1_single_agent.logger import RunLogger
from agent_lab.stage1_single_agent.permissions import (
    PathNotAllowedError,
    check_write_allowed,
)
from agent_lab.stage2_multi_agent.config import PLAN_RETRIES, PLANNER_MAX_TURNS
from agent_lab.stage2_multi_agent.llm_loop import run_role_loop


class PlanError(Exception):
    """planner 沒有產生合格的計畫時丟出。"""


PLANNER_PROMPT = """你是 planner，負責為「修正 Python 專案的 ruff 問題」擬定計畫。
你不能修改任何檔案，修改由別的角色負責。
可用工具：list_dir、read_file、run_command（只能執行 ruff check 與 pytest）。
路徑一律使用相對於專案根目錄的相對路徑。

工作步驟：
1. 用 list_dir 了解專案結構。
2. 用 read_file 讀 ruff.toml，確認啟用的規則與行長度上限。
3. 執行 ruff check --output-format=concise .，取得所有問題。
4. 讀取有問題的檔案，確認每個問題實際的內容。
5. 輸出計畫。

計畫規則：
- 以檔案為單位：同一個檔案的所有問題整理在同一筆，讓後續一次改完。
- 只列需要修改的 .py 檔案；不可以列 tests/、ruff.toml、pytest.ini 或其他設定檔。
- 修法必須真正解決問題，不可以建議用 noqa、type: ignore 等註解讓檢查通過。
- 若修法可能產生新的問題（例如字串變長而超過行長度上限），請在 fix 裡寫出避免的做法。
- 沒有任何問題時，files 填空陣列。
- 不要假設內容，必須根據工具實際讀到的結果。

最後一則訊息只輸出一個 JSON 物件，不要有其他文字。格式：
{"summary": "一句話總結", "constraints": ["行長度上限 88"],
 "files": [{"path": "calculator.py",
            "issues": [{"rule": "F401", "line": 1, "fix": "刪除未使用的 import os"}]}]}"""


def extract_json(text: str):
    """從模型輸出取出 JSON；容忍 ``` 程式碼區塊或前後多餘的文字。"""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise PlanError("輸出不是合法的 JSON。")


def _require_str(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{where} 必須是非空字串。")
    return value


def validate_plan(obj) -> dict:
    """檢查計畫格式與內容；任何不合格都丟 PlanError（訊息會回饋給 planner）。"""
    if not isinstance(obj, dict):
        raise PlanError("最外層必須是 JSON 物件。")
    summary = _require_str(obj.get("summary"), "summary")

    constraints = obj.get("constraints", [])
    if not isinstance(constraints, list) or not all(isinstance(c, str) for c in constraints):
        raise PlanError("constraints 必須是字串陣列。")

    files = obj.get("files")
    if not isinstance(files, list):
        raise PlanError("files 必須是陣列（沒有問題時用空陣列）。")

    seen = set()
    for i, item in enumerate(files):
        where = f"files[{i}]"
        if not isinstance(item, dict):
            raise PlanError(f"{where} 必須是物件。")
        path = _require_str(item.get("path"), f"{where}.path")
        try:
            target = check_write_allowed(path)  # 存在、是 .py、不是受保護檔案
        except PathNotAllowedError as e:
            raise PlanError(f"{where}.path：{e}") from e
        if target in seen:
            raise PlanError(f"{where}.path：'{path}' 重複，同一個檔案的問題請合併成一筆。")
        seen.add(target)

        issues = item.get("issues")
        if not isinstance(issues, list) or not issues:
            raise PlanError(f"{where}.issues 必須是非空陣列。")
        for j, issue in enumerate(issues):
            iw = f"{where}.issues[{j}]"
            if not isinstance(issue, dict):
                raise PlanError(f"{iw} 必須是物件。")
            _require_str(issue.get("rule"), f"{iw}.rule")
            _require_str(issue.get("fix"), f"{iw}.fix")
            line = issue.get("line")
            if line is not None and (isinstance(line, bool) or not isinstance(line, int)):
                raise PlanError(f"{iw}.line 必須是整數或省略。")

    return {"summary": summary, "constraints": constraints, "files": files}


def plan_task(task: str, logger: RunLogger) -> dict:
    """執行 planner；計畫無效時帶著錯誤原因重試，仍無效就丟 PlanError。

    回傳 {"plan", "tokens", "attempts"}。
    """
    totals = {"prompt": 0, "completion": 0, "total": 0}
    user_message = task
    error = "未知錯誤"

    for attempt in range(1, PLAN_RETRIES + 2):
        result = run_role_loop("planner", PLANNER_PROMPT, user_message, PLANNER_MAX_TURNS, logger)
        for k in totals:
            totals[k] += result["tokens"][k]

        try:
            if result["reason"] == "max_turns":
                raise PlanError(f"planner 在 {PLANNER_MAX_TURNS} 輪內沒有給出計畫。")
            plan = validate_plan(extract_json(result["text"]))
        except PlanError as e:
            error = str(e)
            logger.log("plan_invalid", attempt=attempt, error=error, output=result["text"])
            print(f"[planner] 第 {attempt} 次的計畫無效：{error}")
            user_message = (
                f"{task}\n\n"
                f"你上一次輸出的計畫無效，原因：{error}\n"
                f"上一次的輸出：{result['text'][:2000]}\n"
                "請修正後，重新輸出完整的 JSON 物件。"
            )
            continue

        logger.log("plan", attempt=attempt, plan=plan)
        return {"plan": plan, "tokens": totals, "attempts": attempt}

    raise PlanError(f"重試 {PLAN_RETRIES} 次後計畫仍然無效：{error}")


def main() -> None:
    task = " ".join(sys.argv[1:]) or "找出專案裡所有 ruff 問題，並擬定修正計畫"
    logger = RunLogger()
    if logger.path:
        print(f"日誌：{logger.path}")
    try:
        out = plan_task(task, logger)
    except (PlanError, ModelCallError, RuntimeError) as e:
        print(f"執行失敗：{e}")
        sys.exit(1)
    print("\n=== 計畫 ===")
    print(json.dumps(out["plan"], ensure_ascii=False, indent=2))
    t = out["tokens"]
    print(f"\n嘗試次數：{out['attempts']}｜token：{t['total']}"
          f"（prompt {t['prompt']} + completion {t['completion']}）")


if __name__ == "__main__":
    main()