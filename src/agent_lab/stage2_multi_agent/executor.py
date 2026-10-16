import json
import sys

from agent_lab.stage1_single_agent.agent import ModelCallError
from agent_lab.stage1_single_agent.logger import RunLogger
from agent_lab.stage1_single_agent.permissions import resolve_safe_path
from agent_lab.stage2_multi_agent.config import EXECUTOR_MAX_TURNS
from agent_lab.stage2_multi_agent.llm_loop import run_role_loop
from agent_lab.stage2_multi_agent.planner import PlanError, plan_task

# 這三個開頭必須和階段一 tools.write_file 的回傳訊息一致（改一邊要改另一邊）
WRITTEN_PREFIX = "已寫入"
REJECTED_PREFIX = "使用者拒絕"
UNCHANGED_PREFIX = "內容沒有變動"

FEEDBACK_MAX_CHARS = 1500

EXECUTOR_PROMPT = """你是 executor，負責依照計畫修正「一個」Python 檔案。
可用工具：list_dir、read_file、run_command（只能執行 ruff check 與 pytest）、write_file。
路徑一律使用相對於專案根目錄的相對路徑。你只能修改指定的目標檔案。

工作步驟：
1. 用 read_file 讀目標檔案。
2. 依照計畫修正所有列出的問題，並在同一次修改中一起處理。
3. 用 write_file 寫入「修改後的完整檔案內容」，不可省略沒改的部分。盡量只寫入一次；
   每次寫入使用者都會審核，可能被拒絕。
4. 寫入後用 run_command 執行 ruff check --output-format=concise <目標檔案> 確認。
   若有你造成或遺漏的問題，最多再修正一次。
5. 結束時用一兩句話說明你做了什麼。

規則：
- 不可以用 noqa、type: ignore 等註解讓檢查通過，必須真正修正問題。
- 不要改變程式行為：保留所有函式名稱、參數與回傳值，讓既有測試仍然通過。
- 修改後的每一行都要符合計畫中的限制（例如行長度上限）。
- 不可以修改 tests/、ruff.toml、pytest.ini 或其他檔案。"""


def build_file_message(item: dict, plan: dict, feedback: str) -> str:
    lines = [f"目標檔案：{item['path']}", f"計畫總結：{plan['summary']}"]
    if plan["constraints"]:
        lines.append("限制：")
        lines += [f"- {c}" for c in plan["constraints"]]
    lines.append("要修正的問題：")
    for issue in item["issues"]:
        where = f"第 {issue['line']} 行" if issue.get("line") is not None else "位置不明"
        lines.append(f"- [{issue['rule']}] {where}：{issue['fix']}")
    if feedback:
        lines.append("")
        lines.append("上一輪檢查的結果顯示仍有問題，請特別處理（以實際讀到的檔案為準）：")
        lines.append(feedback[:FEEDBACK_MAX_CHARS])
    return "\n".join(lines)


def _make_hook(counts: dict):
    """記錄 write_file 的結果；使用者拒絕時要求立刻停止迴圈。"""

    def hook(name: str, arguments: dict, result: str) -> bool:
        if name != "write_file":
            return False
        if result.startswith(WRITTEN_PREFIX):
            counts["written"] += 1
        elif result.startswith(UNCHANGED_PREFIX):
            counts["unchanged"] += 1
        elif result.startswith(REJECTED_PREFIX):
            counts["rejected"] += 1
            return True
        return False

    return hook


def execute_plan(plan: dict, logger: RunLogger, feedback: str = "") -> dict:
    """依計畫逐檔修改。回傳 {"files", "tokens", "stopped_by_user"}。

    每個檔案的 status：written／unchanged／no_write／max_turns／rejected_by_user／skipped。
    使用者按 n 視為人的決定：立刻停止，後面的檔案標為 skipped。
    """
    totals = {"prompt": 0, "completion": 0, "total": 0}
    results = []
    stopped = False

    for item in plan["files"]:
        path = item["path"]
        if stopped:
            results.append({"path": path, "status": "skipped", "writes": 0,
                            "turns": 0, "tokens": dict.fromkeys(totals, 0)})
            continue

        counts = {"written": 0, "unchanged": 0, "rejected": 0}
        out = run_role_loop(
            "executor",
            EXECUTOR_PROMPT,
            build_file_message(item, plan, feedback),
            EXECUTOR_MAX_TURNS,
            logger,
            write_scope={resolve_safe_path(path)},
            on_tool_result=_make_hook(counts),
        )
        for k in totals:
            totals[k] += out["tokens"][k]

        if counts["rejected"]:
            status, stopped = "rejected_by_user", True
        elif counts["written"]:
            status = "written"
        elif counts["unchanged"]:
            status = "unchanged"
        elif out["reason"] == "max_turns":
            status = "max_turns"
        else:
            status = "no_write"

        entry = {"path": path, "status": status, "writes": counts["written"],
                 "turns": out["turns"], "tokens": out["tokens"]}
        results.append(entry)
        logger.log("executor_file", **entry)

    return {"files": results, "tokens": totals, "stopped_by_user": stopped}


def main() -> None:
    task = " ".join(sys.argv[1:]) or "找出專案裡所有 ruff 問題，並擬定修正計畫"
    logger = RunLogger()
    if logger.path:
        print(f"日誌：{logger.path}")
    try:
        planned = plan_task(task, logger)
        plan = planned["plan"]
        print(f"\n[計畫] {plan['summary']}（{len(plan['files'])} 個檔案）\n")
        out = execute_plan(plan, logger)
    except (PlanError, ModelCallError, RuntimeError) as e:
        print(f"執行失敗：{e}")
        sys.exit(1)

    print("\n=== executor 結果 ===")
    print(json.dumps(out["files"], ensure_ascii=False, indent=2))
    t = out["tokens"]
    print(f"executor token：{t['total']}（prompt {t['prompt']} + completion {t['completion']}）")
    if out["stopped_by_user"]:
        print("使用者拒絕了修改，已停止。")
    print("請自行執行 ruff check . 與 pytest -q 確認結果（reviewer 在第 5 步才會做）。")


if __name__ == "__main__":
    main()