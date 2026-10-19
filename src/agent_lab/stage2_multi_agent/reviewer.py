import hashlib
import re
import sys

from agent_lab.config import PROJECT_DIR
from agent_lab.stage1_single_agent.logger import RunLogger
from agent_lab.stage1_single_agent.permissions import (
    ContentNotAllowedError,
    check_no_new_suppression,
    resolve_safe_path,
)
from agent_lab.stage1_single_agent.tools import run_command

# 這些是工具自己產生的快取，不算「專案檔案被改」
_IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
MAX_TEXT_BYTES = 200_000       # 超過這個大小的 .py 檔不保留全文（只比對雜湊）
DETAIL_MAX_CHARS = 800         # 每項檢查放進回饋的細節長度上限
_EXIT_RE = re.compile(r"^\[exit code (-?\d+)\]")


def take_snapshot() -> dict:
    """記錄專案內每個檔案的雜湊值；.py 檔另存全文，用來比對新增的抑制註解。"""
    snap = {}
    for path in sorted(PROJECT_DIR.rglob("*")):
        rel = path.relative_to(PROJECT_DIR)
        if not path.is_file() or _IGNORED_DIRS & set(rel.parts):
            continue
        data = path.read_bytes()
        keep_text = path.suffix == ".py" and len(data) <= MAX_TEXT_BYTES
        snap[rel.as_posix()] = {
            "sha": hashlib.sha256(data).hexdigest(),
            "text": data.decode("utf-8", errors="replace") if keep_text else None,
        }
    return snap


def _planned_paths(plan: dict) -> set[str]:
    return {
        resolve_safe_path(item["path"]).relative_to(PROJECT_DIR).as_posix()
        for item in plan["files"]
    }


def check_integrity(baseline: dict, current: dict, allowed: set[str]) -> list[str]:
    """除了計畫列出的檔案，專案內不應有任何檔案被新增、刪除或修改。"""
    problems = []
    for rel in sorted(set(baseline) | set(current)):
        if rel not in current:
            problems.append(f"{rel} 被刪除")
        elif rel not in baseline:
            problems.append(f"{rel} 是新增的檔案")
        elif baseline[rel]["sha"] != current[rel]["sha"] and rel not in allowed:
            problems.append(f"{rel} 被修改（不在計畫內）")
    return problems


def check_suppressions(baseline: dict, current: dict, allowed: set[str]) -> list[str]:
    """計畫內被修改的 .py 檔，不可以比原本多出 noqa、type: ignore 等註解。"""
    problems = []
    for rel in sorted(allowed):
        old, new = baseline.get(rel), current.get(rel)
        if not old or not new or old["sha"] == new["sha"]:
            continue
        if old["text"] is None or new["text"] is None:
            continue
        try:
            check_no_new_suppression(old["text"], new["text"])
        except ContentNotAllowedError:
            problems.append(f"{rel} 新增了抑制檢查的註解（noqa、type: ignore 等）")
    return problems


def _run(command: list) -> tuple[int | None, str]:
    """執行白名單指令，回傳 (exit code, 輸出)；解析不到 exit code 時第一個值是 None。"""
    out = run_command(command)
    match = _EXIT_RE.match(out)
    if not match:
        return None, out
    return int(match.group(1)), out[match.end():].strip()


def _command_check(name: str, command: list, tail: bool = False) -> dict:
    code, output = _run(command)
    if code == 0:
        return {"name": name, "passed": True, "detail": ""}
    if code is None:
        detail = f"無法取得 exit code：{output}"
    else:
        detail = f"exit code {code}\n{output}"
    detail = detail[-DETAIL_MAX_CHARS:] if tail else detail[:DETAIL_MAX_CHARS]
    return {"name": name, "passed": False, "detail": detail}


def review(plan: dict, baseline: dict | None, logger: RunLogger) -> dict:
    """檢查修改結果，全部由程式判定（不呼叫模型）。

    回傳 {"passed", "action", "checks", "feedback"}；action 是 done／retry／abort。
    baseline 為 None 時略過完整性與註解檢查（只用在手動測試）。
    """
    checks = []
    action = "done"

    if baseline is not None:
        allowed = _planned_paths(plan)
        current = take_snapshot()
        integrity = check_integrity(baseline, current, allowed)
        checks.append({"name": "檔案完整性", "passed": not integrity,
                       "detail": "；".join(integrity)})
        if integrity:
            # 有不該動的檔案被動過：不再執行專案程式碼（ruff 設定或測試可能已被改），直接中止
            return _finish(checks, "abort", logger)
        suppress = check_suppressions(baseline, current, allowed)
        checks.append({"name": "沒有新增抑制註解", "passed": not suppress,
                       "detail": "；".join(suppress)})

    checks.append(_command_check("ruff check", ["ruff", "check", "--output-format=concise", "."]))
    checks.append(_command_check("pytest", ["pytest", "-q", "--tb=short"], tail=True))

    if not all(c["passed"] for c in checks):
        action = "retry"
    return _finish(checks, action, logger)


def _finish(checks: list, action: str, logger: RunLogger) -> dict:
    failed = [c for c in checks if not c["passed"]]
    feedback = "\n\n".join(f"[{c['name']}] {c['detail']}" for c in failed)
    verdict = {"passed": not failed, "action": action, "checks": checks, "feedback": feedback}
    logger.log("review", passed=verdict["passed"], action=action,
               checks=[{"name": c["name"], "passed": c["passed"], "detail": c["detail"][:500]}
                       for c in checks])
    return verdict


def main() -> None:
    """手動測試：檢查目前的專案狀態（沒有基準快照，只跑 ruff 與 pytest）。"""
    logger = RunLogger()
    verdict = review({"files": []}, None, logger)
    print("（無基準快照，略過完整性與註解檢查）")
    for c in verdict["checks"]:
        print(f"{'通過' if c['passed'] else '失敗'}｜{c['name']}")
    print(f"\n判定：{'通過' if verdict['passed'] else '未通過'}｜下一步：{verdict['action']}")
    if verdict["feedback"]:
        print("\n--- 給 executor 的回饋 ---")
        print(verdict["feedback"])
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()