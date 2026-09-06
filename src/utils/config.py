"""Centralised configuration. Every tunable lives here, sourced from env vars / .env —
nothing else in the codebase should read os.environ directly."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    data_dir: str = "./data"
    models_dir: str = "./models"
    artifacts_dir: str = "./artifacts"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "credit_risk"
    postgres_user: str = "credit_risk_app"
    postgres_password: str = "change_me"

    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    demo_mode: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def resolve(self, path_str: str) -> Path:
        """Resolve a configured path relative to the project root when it's relative,
        so behaviour is identical whether run from the repo root or from a subfolder."""
        path = Path(path_str)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def data_path(self) -> Path:
        return self.resolve(self.data_dir)

    @property
    def models_path(self) -> Path:
        return self.resolve(self.models_dir)

    @property
    def artifacts_path(self) -> Path:
        return self.resolve(self.artifacts_dir)


settings = Settings()
