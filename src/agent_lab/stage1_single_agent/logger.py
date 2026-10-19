import json
import os
from datetime import datetime
from pathlib import Path

from agent_lab.config import API_KEY_ENV_VAR
from agent_lab.stage1_single_agent.config import LOG_DIR


class RunLogger:
    """每次執行寫一個 JSONL 檔，一行一個事件。寫入失敗只警告，不中斷 agent。"""

    def __init__(self, log_dir: Path | None = None) -> None:
        self.path = None
        self._warned = False
        try:
            directory = log_dir or LOG_DIR
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = directory / f"run_{stamp}.jsonl"
            n = 1
            while path.exists():  # 同一秒內重複執行時避免寫進同一個檔
                n += 1
                path = directory / f"run_{stamp}_{n}.jsonl"
            self.path = path
        except OSError as e:
            self._warn(f"無法建立日誌資料夾：{e}")

    def _warn(self, message: str) -> None:
        if not self._warned:  # 只警告一次，避免洗版
            print(f"[警告] 日誌功能停用：{message}")
            self._warned = True

    @staticmethod
    def _redact(text: str) -> str:
        """保險：萬一金鑰字串出現在任何內容裡，寫入前先遮蔽。"""
        key = os.getenv(API_KEY_ENV_VAR)
        if key:
            text = text.replace(key, "***").replace(json.dumps(key)[1:-1], "***")
        return text

    def log(self, event: str, **fields) -> None:
        if self.path is None:
            return
        record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event}
        record.update(fields)
        try:
            line = self._redact(json.dumps(record, ensure_ascii=False, default=str))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            self._warn(f"寫入失敗：{e}")