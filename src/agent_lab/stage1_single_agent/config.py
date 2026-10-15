from pathlib import Path

MAX_TURNS = 15                # 起始值，跑過後再調
MAX_TOOL_OUTPUT_CHARS = 4000  # 起始值，單次工具輸出超過就截斷
COMMAND_TIMEOUT_SECONDS = 30  # run_command 的逾時
LOG_DIR = Path(__file__).resolve().parent / "logs"