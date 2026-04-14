# app/core/config.py
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()  # 自动加载根目录 .env

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "http://127.0.0.1:42060/v1"
    OPENAI_MODEL: str = "openai/qwen3.5-plus"
    OPENAI_TWO_PASS: bool = True
    SCREENING_ALLOW_MOCK: bool = False
    SILICONFLOW_API_KEY: str = ""
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"
    DATABASE_URL: str = "sqlite:///fshd.db"
    STORAGE_TYPE: str = "local"
    STORAGE_LOCAL_DIR: str = "./uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
