import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import uuid
from typing import List

from fastapi import FastAPI, APIRouter
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict

# -----------------------------
# Load environment
# -----------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# IMPORTANT:
# Playwright needs PLAYWRIGHT_BROWSERS_PATH set BEFORE any Playwright import.
# In this container we mount browsers at /pw-browsers.
# On Windows/local setups, forcing /pw-browsers breaks Playwright's default cache
# location and causes: "Executable doesn't exist at \\pw-browsers\\...".
# So we only set it when /pw-browsers exists (container).
if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and os.path.exists("/pw-browsers"):
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/pw-browsers"

MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# -----------------------------
# MongoDB setup
# -----------------------------
mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client[DB_NAME]

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# API Router
# -----------------------------
api_router = APIRouter(prefix="/api")


# Status Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


# Status routes
@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_obj = StatusCheck(**input.model_dump())
    doc = status_obj.model_dump()
    doc["timestamp"] = doc["timestamp"].isoformat()
    await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in checks:
        if isinstance(check.get("timestamp"), str):
            check["timestamp"] = datetime.fromisoformat(check["timestamp"])
    return checks


# -----------------------------
# Lifespan: cleanup only
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Windows Proactor loop for subprocess (Playwright subprocesses)
    import sys
    import asyncio

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    logger.info("Backend startup complete (Playwright launches lazily on /api/browser/session).")
    yield

    # Cleanup Playwright sessions + Mongo
    try:
        from routes.browser import session_manager

        await session_manager.cleanup()
        logger.info("Playwright sessions cleaned up.")
    except Exception as e:
        logger.warning(f"Playwright cleanup skipped/failed: {e}")

    mongo_client.close()
    logger.info("MongoDB client closed.")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Include routers
app.include_router(api_router)

# Include browser, extensions, and search routers
from routes.browser import router as browser_router
from routes.extensions import router as extensions_router
from routes.search import router as search_router

app.include_router(browser_router, prefix="/api")
app.include_router(extensions_router, prefix="/api")
app.include_router(search_router, prefix="/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
