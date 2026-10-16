import json
import sys

from agent_lab.stage1_single_agent.config import LOG_DIR


def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(LOG_DIR.glob("run_*.jsonl"))
        if not files:
            print("logs/ 裡沒有日誌檔。")
            return
        path = files[-1]  # 檔名含時間，排序後最後一個就是最新的
    print(f"日誌：{path}")

    calls = {}  # 輪數 -> 該輪的工具名稱
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    for e in events:
        if e["event"] == "tool_result":
            calls.setdefault(e["turn"], []).append(e["name"])

    print(f"{'輪':>3} {'prompt':>7} {'compl.':>7} {'total':>7}  工具")
    for e in events:
        if e["event"] == "model_response":
            u = e.get("usage", {})
            tools = ", ".join(calls.get(e["turn"], [])) or "(無，最終回答)"
            print(f"{e['turn']:>3} {u.get('prompt', 0):>7} {u.get('completion', 0):>7} "
                  f"{u.get('total', 0):>7}  {tools}")
    for e in events:
        if e["event"] == "run_end":
            t = e["tokens"]
            print(f"結束原因：{e['reason']}｜輪數：{e['turns']}｜"
                  f"token 合計：{t['total']}（prompt {t['prompt']} + completion {t['completion']}）")
            if e.get("error"):
                print(f"錯誤：{e['error']}")


if __name__ == "__main__":
    main()