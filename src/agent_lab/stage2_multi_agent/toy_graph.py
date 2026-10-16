from typing import TypedDict

from langgraph.graph import END, START, StateGraph

MAX_ATTEMPTS = 3


class State(TypedDict):
    pass_at: int          # 模擬用：executor 第幾次嘗試才會修好
    attempt: int          # executor 已嘗試幾次
    passed: bool          # reviewer 的判定
    feedback: str         # reviewer 給 executor 的失敗原因
    history: list[str]    # 過程紀錄，方便觀察


def planner(state: State) -> dict:
    return {"history": state["history"] + ["planner：擬定計畫"]}


def executor(state: State) -> dict:
    n = state["attempt"] + 1
    note = f"executor：第 {n} 次嘗試（收到的回饋：{state['feedback'] or '無'}）"
    return {"attempt": n, "history": state["history"] + [note]}


def reviewer(state: State) -> dict:
    ok = state["attempt"] >= state["pass_at"]
    feedback = "" if ok else f"第 {state['attempt']} 次還有錯誤"
    note = f"reviewer：{'通過' if ok else '失敗'}"
    return {"passed": ok, "feedback": feedback, "history": state["history"] + [note]}


def route_after_review(state: State) -> str:
    if state["passed"]:
        return "done"
    if state["attempt"] >= MAX_ATTEMPTS:
        return "give_up"
    return "retry"


def build():
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"done": END, "give_up": END, "retry": "executor"},
    )
    return g.compile()


def main() -> None:
    app = build()
    for label, pass_at in [("第 2 次修好", 2), ("永遠修不好", 99)]:
        final = app.invoke(
            {"pass_at": pass_at, "attempt": 0, "passed": False, "feedback": "", "history": []}
        )
        print(f"--- {label}：passed={final['passed']}，attempt={final['attempt']}")
        for line in final["history"]:
            print("  ", line)


if __name__ == "__main__":
    main()