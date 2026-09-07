"""FastAPI app tying the ML pipeline, explainability, rules, and chatbot together behind
one HTTP API for the frontend.

The model and training data are loaded once, at import/startup time (via the lru_cache on
load_artifacts()/load_application_train()), not per request — see src/ml/predict.py and
src/data/loader.py for why that matters.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import chat, eda, explain, health, predict, rules
from src.data.loader import load_application_train, table_path
from src.ml.predict import load_artifacts
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Credit Risk Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the EDA notebook's saved charts directly — the API doesn't regenerate them.
app.mount("/charts", StaticFiles(directory=str(settings.artifacts_path / "eda")), name="charts")

app.include_router(health.router)
app.include_router(eda.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(rules.router)
app.include_router(chat.router)


@app.on_event("startup")
def warm_caches():
    logger.info("Warming caches: model artifacts...")
    load_artifacts()
    if table_path("application_train").exists():
        load_application_train()
        logger.info("application_train warmed from local CSV.")
    else:
        # No local CSV on a free-tier host means no RAM to hold all 307k rows resident
        # either — per-request code queries Postgres directly instead (see loader.py).
        logger.info("No local CSV — application_train will be queried from Postgres per request.")
    logger.info("Ready.")
