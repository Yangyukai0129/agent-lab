import sys

from agent_lab.stage1_single_agent.agent import ModelCallError
from agent_lab.stage1_single_agent.logger import RunLogger
from agent_lab.stage2_multi_agent.config import LOG_DIR
from agent_lab.stage2_multi_agent.graph import FINAL_STATUSES, add_tokens, run_graph


def print_report(final: dict) -> None:
    print("\n=== 最終報告 ===")
    print(f"狀態：{final['status']}（{FINAL_STATUSES.get(final['status'], '未知')}）")
    print(f"executor 執行次數：{final['attempt']}")
    print("\n過程：")
    for line in final["history"]:
        print(f"  - {line}")
    if final["executor_runs"]:
        print("\n最後一次 executor 的各檔案結果：")
        for f in final["executor_runs"][-1]["files"]:
            print(f"  - {f['path']}：{f['status']}（寫入 {f['writes']} 次）")
    if final["verdict"]:
        print("\nreviewer 的檢查：")
        for c in final["verdict"]["checks"]:
            print(f"  - {'通過' if c['passed'] else '失敗'}｜{c['name']}")
    t = final["tokens"]
    total = add_tokens(t["planner"], t["executor"])
    print(f"\ntoken：planner {t['planner']['total']}｜executor {t['executor']['total']}"
          f"｜合計 {total['total']}（prompt {total['prompt']} + completion {total['completion']}）")
    if final["status"] != "done" and final["feedback"]:
        print(f"\n未解決的問題：\n{final['feedback']}")


def main() -> None:
    task = " ".join(sys.argv[1:]) or "找出專案裡所有 ruff 問題，並擬定修正計畫並修正"
    logger = RunLogger(LOG_DIR)
    if logger.path:
        print(f"日誌：{logger.path}")
    try:
        final = run_graph(task, logger)
    except (ModelCallError, RuntimeError) as e:
        print(f"\n執行失敗（系統錯誤）：{e}")
        sys.exit(1)
    print_report(final)
    sys.exit(0 if final["status"] == "done" else 1)


if __name__ == "__main__":
    main()