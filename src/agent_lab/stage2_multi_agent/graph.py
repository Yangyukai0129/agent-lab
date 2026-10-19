import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_lab.config import MODEL
from agent_lab.stage1_single_agent.logger import RunLogger
from agent_lab.stage2_multi_agent.config import MAX_ITERATIONS
from agent_lab.stage2_multi_agent.executor import execute_plan
from agent_lab.stage2_multi_agent.planner import PlanError, plan_task
from agent_lab.stage2_multi_agent.reviewer import review, take_snapshot

# status 為空字串代表流程還在進行；其餘都是最終狀態
FINAL_STATUSES = {
    "done": "成功：ruff 與 pytest 都通過",
    "max_iterations": "未修好：已達重試上限",
    "aborted": "中止：偵測到不該被修改的檔案變動",
    "rejected_by_user": "停止：使用者拒絕了修改",
    "plan_failed": "失敗：planner 沒有產生合格的計畫",
    "plan_mismatch": "異常：計畫說沒有問題，但檢查沒有通過",
}


def zero_tokens() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0}


def add_tokens(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


class State(TypedDict):
    task: str
    plan: dict
    baseline: dict          # 開始時的專案快照（reviewer 的比對基準）
    attempt: int            # executor 已執行幾次
    feedback: str           # reviewer 給 executor 的失敗原因
    verdict: dict
    status: str
    tokens: dict            # {"planner": {...}, "executor": {...}}
    # 加了 reducer 的欄位：節點只要回傳「這次新增的項目」，LangGraph 會自動接在後面
    executor_runs: Annotated[list, operator.add]
    history: Annotated[list, operator.add]


def initial_state(task: str) -> State:
    return {
        "task": task, "plan": {}, "baseline": {}, "attempt": 0, "feedback": "",
        "verdict": {}, "status": "",
        "tokens": {"planner": zero_tokens(), "executor": zero_tokens()},
        "executor_runs": [], "history": [],
    }


def build_graph(logger: RunLogger):
    def snapshot_node(state: State) -> dict:
        print("\n===== snapshot =====")
        baseline = take_snapshot()
        return {"baseline": baseline,
                "history": [f"snapshot：記錄 {len(baseline)} 個檔案的快照"]}

    def planner_node(state: State) -> dict:
        print("\n===== planner =====")
        try:
            out = plan_task(state["task"], logger)
        except PlanError as e:
            spent = e.tokens or zero_tokens()
            return {"status": "plan_failed", "feedback": str(e),
                    "tokens": {**state["tokens"], "planner": spent},
                    "history": [f"planner：失敗：{e}"]}
        plan = out["plan"]
        return {"plan": plan,
                "tokens": {**state["tokens"], "planner": out["tokens"]},
                "history": [f"planner：計畫 {len(plan['files'])} 個檔案（嘗試 {out['attempts']} 次）"]}

    def executor_node(state: State) -> dict:
        attempt = state["attempt"] + 1
        print(f"\n===== executor（第 {attempt} 次） =====")
        out = execute_plan(state["plan"], logger, state["feedback"])
        summary = "、".join(f"{f['path']}={f['status']}" for f in out["files"])
        tokens = {**state["tokens"],
                  "executor": add_tokens(state["tokens"]["executor"], out["tokens"])}
        return {
            "attempt": attempt,
            "executor_runs": [{"attempt": attempt, **out}],
            "tokens": tokens,
            "status": "rejected_by_user" if out["stopped_by_user"] else "",
            "history": [f"executor：第 {attempt} 次：{summary}"],
        }

    def reviewer_node(state: State) -> dict:
        print("\n===== reviewer =====")
        verdict = review(state["plan"], state["baseline"], logger)
        action = verdict["action"]
        if action == "done":
            status = "done"
        elif action == "abort":
            status = "aborted"
        elif not state["plan"]["files"]:
            status = "plan_mismatch"   # 沒有計畫卻沒通過：重試也沒有東西可以改
        elif state["attempt"] >= MAX_ITERATIONS:
            status = "max_iterations"
        else:
            status = ""                # 繼續重試
        failed = [c["name"] for c in verdict["checks"] if not c["passed"]]
        note = "通過" if verdict["passed"] else f"未通過（{'、'.join(failed)}）"
        return {"verdict": verdict, "feedback": verdict["feedback"], "status": status,
                "history": [f"reviewer：{note} → {action}"]}

    def route_after_planner(state: State) -> str:
        if state["status"] == "plan_failed":
            return "end"
        return "execute" if state["plan"]["files"] else "review"

    def route_after_executor(state: State) -> str:
        return "end" if state["status"] else "review"

    def route_after_reviewer(state: State) -> str:
        return "end" if state["status"] else "retry"

    g = StateGraph(State)
    g.add_node("snapshot", snapshot_node)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("reviewer", reviewer_node)
    g.add_edge(START, "snapshot")
    g.add_edge("snapshot", "planner")
    g.add_conditional_edges(
        "planner", route_after_planner,
        {"execute": "executor", "review": "reviewer", "end": END},
    )
    g.add_conditional_edges(
        "executor", route_after_executor, {"review": "reviewer", "end": END},
    )
    g.add_conditional_edges(
        "reviewer", route_after_reviewer, {"retry": "executor", "end": END},
    )
    return g.compile()


def run_graph(task: str, logger: RunLogger) -> State:
    """執行完整流程；不管成功、失敗或被中斷，都會寫入 run_end。"""
    logger.log("run_start", task=task, model=MODEL, max_iterations=MAX_ITERATIONS)
    app = build_graph(logger)
    final, error = None, None
    try:
        final = app.invoke(initial_state(task))
        return final
    except BaseException as e:  # 包含 Ctrl+C
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        if final:
            tokens = final["tokens"]
            logger.log("run_end", status=final["status"], attempts=final["attempt"],
                       tokens={n: tokens[n] for n in tokens},
                       total=add_tokens(tokens["planner"], tokens["executor"]), error=None)
        else:
            logger.log("run_end", status="error", attempts=None, tokens=None, total=None,
                       error=error)