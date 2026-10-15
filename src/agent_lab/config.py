import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_DIR = Path(__file__).resolve().parent / "sample_project"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"  # 只放環境變數的「名稱」
MODEL = "nvidia/nemotron-3.5-lightning:free"  # TODO: 到 openrouter.ai/models 選一個支援 tools 的模型

def get_api_key() -> str:
    key = os.getenv(API_KEY_ENV_VAR)
    if not key:
        raise RuntimeError(f"找不到環境變數 {API_KEY_ENV_VAR}，請確認 .env 內容")
    return key