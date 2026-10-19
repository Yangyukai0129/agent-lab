import json
import sys

from agent_lab.stage2_multi_agent.config import LOG_DIR


def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(LOG_DIR.glob("run_*.jsonl"))
        if not files:
            print("logs/ 裡沒有日誌檔。")
            return
        path = files[-1]
    print(f"日誌：{path}")
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    nodes = {}
    for e in events:
        if e["event"] == "model_response":
            n = nodes.setdefault(e["node"], {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
            n["calls"] += 1
            for k in ("prompt", "completion", "total"):
                n[k] += e["usage"][k]
    print(f"\n{'節點':<10} {'模型呼叫':>8} {'prompt':>8} {'compl.':>8} {'total':>8}")
    for name, n in nodes.items():
        print(f"{name:<10} {n['calls']:>8} {n['prompt']:>8} {n['completion']:>8} {n['total']:>8}")

    print("\nexecutor 各檔案：")
    for e in events:
        if e["event"] == "executor_file":
            print(f"  {e['path']}：{e['status']}（寫入 {e['writes']} 次，{e['turns']} 輪，"
                  f"{e['tokens']['total']} tokens）")
    print("\nreviewer 判定（依序）：")
    for i, e in enumerate((e for e in events if e["event"] == "review"), 1):
        bad = [c["name"] for c in e["checks"] if not c["passed"]]
        print(f"  第 {i} 次：{'通過' if e['passed'] else '未通過'} → {e['action']}"
              + (f"（{'、'.join(bad)}）" if bad else ""))
    for e in events:
        if e["event"] == "run_end":
            print(f"\n結束：{e['status']}｜executor 次數：{e['attempts']}｜"
                  f"token 合計：{(e['total'] or {}).get('total')}")
            if e.get("error"):
                print(f"錯誤：{e['error']}")


if __name__ == "__main__":
    main()