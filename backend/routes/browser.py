from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import base64
import uuid
import logging
import os
from datetime import datetime, timezone

# IMPORTANT: server.py already conditionally sets PLAYWRIGHT_BROWSERS_PATH.
# Keep this file safe for local/dev environments.
if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and os.path.exists("/pw-browsers"):
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/pw-browsers"

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/browser", tags=["browser"])

# -----------------------------
# Config
# -----------------------------
SESSION_TTL_SECONDS = int(os.environ.get("BROWSER_SESSION_TTL_SECONDS", "1800"))  # 30m
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("BROWSER_SESSION_CLEANUP_INTERVAL_SECONDS", "30"))
VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _now_ts() -> float:
    return _utcnow().timestamp()


# -----------------------------
# Browser session management (multi-tab)
# -----------------------------
class BrowserSessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.playwright = None
        self.browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        async with self._lock:
            if self.playwright is None:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins,site-per-process",
                    ],
                )
                logger.info("Playwright browser initialized")

            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Periodically cleanup expired sessions to prevent memory leaks."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self.cleanup_expired_sessions()
                except Exception as e:
                    logger.warning(f"Session cleanup error: {e}")

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=CLEANUP_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass

    async def cleanup_expired_sessions(self):
        now = _now_ts()
        expired = []
        for sid, s in list(self.sessions.items()):
            last = float(s.get("last_activity_ts") or 0)
            if last and (now - last) > SESSION_TTL_SECONDS:
                expired.append(sid)

        for sid in expired:
            logger.info(f"Cleaning up expired session: {sid}")
            await self.close_session(sid)

    async def create_session(self) -> dict:
        await self.initialize()

        session_id = str(uuid.uuid4())
        context = await self.browser.new_context(viewport=VIEWPORT, user_agent=DEFAULT_UA)

        # Create first tab
        tab_id = str(uuid.uuid4())
        page = await context.new_page()

        self.sessions[session_id] = {
            "context": context,
            "tabs": {
                tab_id: {
                    "page": page,
                    "created_at": _utcnow(),
                    "last_activity_ts": _now_ts(),
                    "history": [],
                    "history_index": -1,
                }
            },
            "active_tab_id": tab_id,
            "created_at": _utcnow(),
            "last_activity_ts": _now_ts(),
        }

        logger.info(f"Created browser session: {session_id} (tab: {tab_id})")
        return {"session_id": session_id, "created_at": self.sessions[session_id]["created_at"], "tab_id": tab_id}

    async def get_session(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)

    def _touch(self, session_id: str, tab_id: Optional[str] = None):
        s = self.sessions.get(session_id)
        if not s:
            return
        now = _now_ts()
        s["last_activity_ts"] = now
        if tab_id and tab_id in s.get("tabs", {}):
            s["tabs"][tab_id]["last_activity_ts"] = now

    async def get_page(self, session_id: str, tab_id: Optional[str] = None) -> Page:
        s = self.sessions.get(session_id)
        if not s:
            raise KeyError("session")
        tid = tab_id or s.get("active_tab_id")
        if not tid or tid not in s.get("tabs", {}):
            raise KeyError("tab")
        return s["tabs"][tid]["page"]

    async def list_tabs(self, session_id: str) -> List[dict]:
        s = self.sessions.get(session_id)
        if not s:
            raise KeyError("session")
        out = []
        for tid, t in s.get("tabs", {}).items():
            page: Page = t["page"]
            try:
                title = await page.title()
            except Exception:
                title = ""
            out.append({
                "tab_id": tid,
                "url": page.url,
                "title": title,
                "is_active": tid == s.get("active_tab_id"),
            })
        return out

    async def create_tab(self, session_id: str, url: Optional[str] = None, make_active: bool = True) -> dict:
        s = self.sessions.get(session_id)
        if not s:
            raise KeyError("session")
        context: BrowserContext = s["context"]
        tab_id = str(uuid.uuid4())
        page = await context.new_page()

        s["tabs"][tab_id] = {
            "page": page,
            "created_at": _utcnow(),
            "last_activity_ts": _now_ts(),
            "history": [],
            "history_index": -1,
        }
        if make_active:
            s["active_tab_id"] = tab_id

        self._touch(session_id, tab_id)

        if url:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    await asyncio.sleep(0.25)
                s["tabs"][tab_id]["history"].append(url)
                s["tabs"][tab_id]["history_index"] = 0
            except Exception as e:
                logger.warning(f"Tab initial navigation failed: {e}")

        try:
            title = await page.title()
        except Exception:
            title = ""

        return {"tab_id": tab_id, "url": page.url, "title": title}

    async def activate_tab(self, session_id: str, tab_id: str):
        s = self.sessions.get(session_id)
        if not s:
            raise KeyError("session")
        if tab_id not in s.get("tabs", {}):
            raise KeyError("tab")
        s["active_tab_id"] = tab_id
        self._touch(session_id, tab_id)

    async def close_tab(self, session_id: str, tab_id: str) -> dict:
        s = self.sessions.get(session_id)
        if not s:
            raise KeyError("session")
        tabs = s.get("tabs", {})
        if tab_id not in tabs:
            raise KeyError("tab")

        page: Page = tabs[tab_id]["page"]
        try:
            await page.close()
        except Exception:
            pass

        del tabs[tab_id]

        # If that was the last tab, close the whole session.
        if not tabs:
            await self.close_session(session_id)
            return {"status": "session_closed"}

        if s.get("active_tab_id") == tab_id:
            s["active_tab_id"] = next(iter(tabs.keys()))

        self._touch(session_id, s.get("active_tab_id"))
        return {"status": "tab_closed", "active_tab_id": s.get("active_tab_id")}

    async def close_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            try:
                await session["context"].close()
            except Exception:
                pass
            del self.sessions[session_id]
            logger.info(f"Closed browser session: {session_id}")

    async def cleanup(self):
        self._shutdown_event.set()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except Exception:
                pass

        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)

        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


session_manager = BrowserSessionManager()


# -----------------------------
# Request/Response models
# -----------------------------
class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    tab_id: Optional[str] = None


class TabInfo(BaseModel):
    tab_id: str
    url: str
    title: str
    is_active: bool = False


class CreateTabRequest(BaseModel):
    url: Optional[str] = None
    make_active: bool = True


class CreateTabResponse(BaseModel):
    tab_id: str
    url: str
    title: str


class NavigateRequest(BaseModel):
    url: str
    tab_id: Optional[str] = None


class ClickRequest(BaseModel):
    x: float
    y: float
    button: str = "left"
    click_count: int = Field(default=1, ge=1, le=3)
    tab_id: Optional[str] = None


class TypeRequest(BaseModel):
    text: str
    tab_id: Optional[str] = None


class KeyPressRequest(BaseModel):
    key: str
    modifiers: Optional[Dict[str, bool]] = None
    tab_id: Optional[str] = None


class ScrollRequest(BaseModel):
    delta_x: float = 0
    delta_y: float = 0
    tab_id: Optional[str] = None


class SessionStatusResponse(BaseModel):
    session_id: str
    active_tab_id: str
    current_url: str
    title: str
    can_go_back: bool
    can_go_forward: bool


class ScreenshotResponse(BaseModel):
    screenshot: str  # base64 encoded
    url: str
    title: str
    tab_id: str


# -----------------------------
# Helpers
# -----------------------------
async def _get_tab_state(session: dict, tab_id: str) -> dict:
    if tab_id not in session.get("tabs", {}):
        raise HTTPException(status_code=404, detail="Tab not found")
    return session["tabs"][tab_id]


async def _page_title_safe(page: Page) -> str:
    try:
        return await page.title()
    except Exception:
        return ""


# -----------------------------
# Endpoints
# -----------------------------
@router.post("/session", response_model=CreateSessionResponse)
async def create_session():
    """Create a new browser session (with an initial real Playwright tab)."""
    try:
        created = await session_manager.create_session()
        return CreateSessionResponse(**created)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Close a browser session."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await session_manager.close_session(session_id)
    return {"status": "closed"}


@router.get("/session/{session_id}/tabs", response_model=List[TabInfo])
async def list_tabs(session_id: str):
    """List tabs for a session."""
    try:
        return [TabInfo(**t) for t in await session_manager.list_tabs(session_id)]
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/session/{session_id}/tabs", response_model=CreateTabResponse)
async def create_tab(session_id: str, request: CreateTabRequest):
    """Create a real tab (Playwright page) within a session."""
    try:
        created = await session_manager.create_tab(session_id, url=request.url, make_active=request.make_active)
        return CreateTabResponse(**created)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/tabs/{tab_id}/activate")
async def activate_tab(session_id: str, tab_id: str):
    """Activate an existing tab."""
    try:
        await session_manager.activate_tab(session_id, tab_id)
        return {"status": "activated", "tab_id": tab_id}
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Session not found" if str(e) == "'session'" else "Tab not found")


@router.delete("/session/{session_id}/tabs/{tab_id}")
async def delete_tab(session_id: str, tab_id: str):
    """Close a tab."""
    try:
        return await session_manager.close_tab(session_id, tab_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Session not found" if str(e) == "'session'" else "Tab not found")


@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str, tab_id: Optional[str] = Query(default=None)):
    """Get current session status."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    effective_tab_id = tab_id or session.get("active_tab_id")
    tab_state = await _get_tab_state(session, effective_tab_id)
    page: Page = tab_state["page"]
    history = tab_state["history"]
    history_index = tab_state["history_index"]

    return SessionStatusResponse(
        session_id=session_id,
        active_tab_id=effective_tab_id,
        current_url=page.url,
        title=await _page_title_safe(page),
        can_go_back=history_index > 0,
        can_go_forward=history_index < len(history) - 1,
    )


@router.post("/{session_id}/navigate")
async def navigate(session_id: str, request: NavigateRequest):
    """Navigate to a URL (active tab by default)."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tab_id = request.tab_id or session.get("active_tab_id")
    tab_state = await _get_tab_state(session, tab_id)
    page: Page = tab_state["page"]

    try:
        session_manager._touch(session_id, tab_id)
        await page.goto(request.url, wait_until="domcontentloaded", timeout=60000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            await asyncio.sleep(0.25)

        # Update history (per-tab)
        tab_state["history"] = tab_state["history"][: tab_state["history_index"] + 1]
        tab_state["history"].append(request.url)
        tab_state["history_index"] = len(tab_state["history"]) - 1

        return {"status": "navigated", "url": page.url, "title": await _page_title_safe(page), "tab_id": tab_id}
    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/back")
async def go_back(session_id: str, tab_id: Optional[str] = Query(default=None)):
    """Go back in history (per-tab)."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tid = tab_id or session.get("active_tab_id")
    tab_state = await _get_tab_state(session, tid)

    if tab_state["history_index"] > 0:
        tab_state["history_index"] -= 1
        page: Page = tab_state["page"]
        await page.go_back()
        session_manager._touch(session_id, tid)
        return {"status": "success", "url": page.url, "tab_id": tid}

    return {"status": "no_history", "tab_id": tid}


@router.post("/{session_id}/forward")
async def go_forward(session_id: str, tab_id: Optional[str] = Query(default=None)):
    """Go forward in history (per-tab)."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tid = tab_id or session.get("active_tab_id")
    tab_state = await _get_tab_state(session, tid)

    if tab_state["history_index"] < len(tab_state["history"]) - 1:
        tab_state["history_index"] += 1
        page: Page = tab_state["page"]
        await page.go_forward()
        session_manager._touch(session_id, tid)
        return {"status": "success", "url": page.url, "tab_id": tid}

    return {"status": "no_forward_history", "tab_id": tid}


@router.post("/{session_id}/refresh")
async def refresh(session_id: str, tab_id: Optional[str] = Query(default=None)):
    """Refresh the page."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tid = tab_id or session.get("active_tab_id")
    tab_state = await _get_tab_state(session, tid)
    page: Page = tab_state["page"]
    await page.reload()
    session_manager._touch(session_id, tid)
    return {"status": "refreshed", "url": page.url, "tab_id": tid}


@router.get("/{session_id}/screenshot", response_model=ScreenshotResponse)
async def get_screenshot(session_id: str, tab_id: Optional[str] = Query(default=None)):
    """Get current page screenshot."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tid = tab_id or session.get("active_tab_id")
    tab_state = await _get_tab_state(session, tid)
    page: Page = tab_state["page"]

    try:
        screenshot_bytes = await page.screenshot(type="jpeg", quality=60)
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        session_manager._touch(session_id, tid)

        return ScreenshotResponse(
            screenshot=f"data:image/jpeg;base64,{screenshot_base64}",
            url=page.url,
            title=await _page_title_safe(page),
            tab_id=tid,
        )
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/click")
async def click(session_id: str, request: ClickRequest):
    """Click at position."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tid = request.tab_id or session.get("active_tab_id")
    page = await session_manager.get_page(session_id, tid)
    await page.mouse.click(request.x, request.y, button=request.button, click_count=request.click_count)
    session_manager._touch(session_id, tid)
    return {"status": "clicked", "tab_id": tid}


@router.post("/{session_id}/type")
async def type_text(session_id: str, request: TypeRequest):
    """Type text."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tid = request.tab_id or session.get("active_tab_id")
    page = await session_manager.get_page(session_id, tid)
    await page.keyboard.type(request.text)
    session_manager._touch(session_id, tid)
    return {"status": "typed", "tab_id": tid}


@router.post("/{session_id}/keypress")
async def keypress(session_id: str, request: KeyPressRequest):
    """Press a key (supports modifiers)."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tid = request.tab_id or session.get("active_tab_id")
    page = await session_manager.get_page(session_id, tid)

    modifiers = request.modifiers or {}
    keys = []

    if modifiers.get("ctrl"):
        keys.append("Control")
    if modifiers.get("alt"):
        keys.append("Alt")
    if modifiers.get("shift"):
        keys.append("Shift")
    if modifiers.get("meta"):
        keys.append("Meta")

    keys.append(request.key)
    await page.keyboard.press("+".join(keys))
    session_manager._touch(session_id, tid)

    return {"status": "pressed", "tab_id": tid}


@router.post("/{session_id}/scroll")
async def scroll(session_id: str, request: ScrollRequest):
    """Scroll the page."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    tid = request.tab_id or session.get("active_tab_id")
    page = await session_manager.get_page(session_id, tid)
    await page.mouse.wheel(request.delta_x, request.delta_y)
    session_manager._touch(session_id, tid)
    return {"status": "scrolled", "tab_id": tid}


# -----------------------------
# WebSocket for real-time streaming (multi-tab)
# -----------------------------
@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    # Client-controlled active tab for this WS connection
    client_tab_id = session.get("active_tab_id")
    streaming = True
    is_navigating = False

    async def send_status():
        nonlocal client_tab_id
        try:
            s = await session_manager.get_session(session_id)
            if not s:
                return
            tid = client_tab_id or s.get("active_tab_id")
            tab_state = s.get("tabs", {}).get(tid)
            if not tab_state:
                return
            page: Page = tab_state["page"]
            history = tab_state["history"]
            history_index = tab_state["history_index"]
            await websocket.send_json({
                "type": "status",
                "tab_id": tid,
                "url": page.url,
                "title": await _page_title_safe(page),
                "can_go_back": history_index > 0,
                "can_go_forward": history_index < len(history) - 1,
            })
        except Exception:
            pass

    async def stream_screenshots():
        nonlocal streaming, is_navigating, client_tab_id
        while streaming:
            try:
                if is_navigating:
                    await asyncio.sleep(0.25)
                    continue

                s = await session_manager.get_session(session_id)
                if not s:
                    await asyncio.sleep(0.5)
                    continue
                tid = client_tab_id or s.get("active_tab_id")
                tab_state = s.get("tabs", {}).get(tid)
                if not tab_state:
                    await asyncio.sleep(0.2)
                    continue

                page: Page = tab_state["page"]
                screenshot_bytes = await page.screenshot(type="jpeg", quality=50)
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

                await websocket.send_json({
                    "type": "screenshot",
                    "tab_id": tid,
                    "data": f"data:image/jpeg;base64,{screenshot_base64}",
                    "url": page.url,
                    "title": await _page_title_safe(page),
                })

                # Send status every few frames
                if int(_now_ts() * 10) % 10 == 0:
                    await send_status()

                await asyncio.sleep(0.15)  # ~7 FPS
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await asyncio.sleep(0.5)

    # Send initial tabs state
    try:
        tabs = await session_manager.list_tabs(session_id)
        await websocket.send_json({"type": "tabs", "tabs": tabs, "active_tab_id": session.get("active_tab_id")})
    except Exception:
        pass

    stream_task = asyncio.create_task(stream_screenshots())

    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("type")

            if event_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif event_type == "switch_tab":
                tab_id = data.get("tab_id")
                if tab_id:
                    try:
                        await session_manager.activate_tab(session_id, tab_id)
                        client_tab_id = tab_id
                        await websocket.send_json({"type": "tab_activated", "tab_id": tab_id})
                        await send_status()
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": str(e)})

            elif event_type == "navigate":
                url = data.get("url")
                tab_id = data.get("tab_id") or client_tab_id
                if url:
                    try:
                        is_navigating = True
                        s = await session_manager.get_session(session_id)
                        if not s:
                            raise RuntimeError("Session closed")
                        if tab_id and tab_id in s.get("tabs", {}):
                            await session_manager.activate_tab(session_id, tab_id)
                            client_tab_id = tab_id
                        tid = client_tab_id or s.get("active_tab_id")
                        tab_state = s["tabs"][tid]
                        page: Page = tab_state["page"]

                        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                        try:
                            await page.wait_for_load_state("networkidle", timeout=10000)
                        except Exception:
                            await asyncio.sleep(0.25)

                        # Update history per-tab
                        tab_state["history"] = tab_state["history"][: tab_state["history_index"] + 1]
                        tab_state["history"].append(url)
                        tab_state["history_index"] = len(tab_state["history"]) - 1
                        session_manager._touch(session_id, tid)
                        await send_status()
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    finally:
                        is_navigating = False

            elif event_type == "click":
                x, y = data.get("x", 0), data.get("y", 0)
                button = data.get("button", "left")
                click_count = int(data.get("clickCount", 1) or 1)
                tab_id = data.get("tab_id") or client_tab_id
                page = await session_manager.get_page(session_id, tab_id)
                await page.mouse.click(x, y, button=button, click_count=click_count)
                session_manager._touch(session_id, tab_id)

            elif event_type == "type":
                text = data.get("text", "")
                tab_id = data.get("tab_id") or client_tab_id
                page = await session_manager.get_page(session_id, tab_id)
                await page.keyboard.type(text)
                session_manager._touch(session_id, tab_id)

            elif event_type == "keypress":
                key = data.get("key", "")
                modifiers = data.get("modifiers") or {}
                tab_id = data.get("tab_id") or client_tab_id
                page = await session_manager.get_page(session_id, tab_id)

                keys = []
                if modifiers.get("ctrl"):
                    keys.append("Control")
                if modifiers.get("alt"):
                    keys.append("Alt")
                if modifiers.get("shift"):
                    keys.append("Shift")
                if modifiers.get("meta"):
                    keys.append("Meta")
                keys.append(key)
                await page.keyboard.press("+".join(keys))
                session_manager._touch(session_id, tab_id)

            elif event_type == "scroll":
                delta_x = data.get("deltaX", 0)
                delta_y = data.get("deltaY", 0)
                tab_id = data.get("tab_id") or client_tab_id
                page = await session_manager.get_page(session_id, tab_id)
                await page.mouse.wheel(delta_x, delta_y)
                session_manager._touch(session_id, tab_id)

            elif event_type == "back":
                s = await session_manager.get_session(session_id)
                if not s:
                    continue
                tid = client_tab_id or s.get("active_tab_id")
                tab_state = s.get("tabs", {}).get(tid)
                if tab_state and tab_state["history_index"] > 0:
                    tab_state["history_index"] -= 1
                    await tab_state["page"].go_back()
                    session_manager._touch(session_id, tid)
                    await send_status()

            elif event_type == "forward":
                s = await session_manager.get_session(session_id)
                if not s:
                    continue
                tid = client_tab_id or s.get("active_tab_id")
                tab_state = s.get("tabs", {}).get(tid)
                if tab_state and tab_state["history_index"] < len(tab_state["history"]) - 1:
                    tab_state["history_index"] += 1
                    await tab_state["page"].go_forward()
                    session_manager._touch(session_id, tid)
                    await send_status()

            elif event_type == "refresh":
                tab_id = data.get("tab_id") or client_tab_id
                page = await session_manager.get_page(session_id, tab_id)
                await page.reload()
                session_manager._touch(session_id, tab_id)
                await send_status()

            elif event_type == "get_tabs":
                tabs = await session_manager.list_tabs(session_id)
                await websocket.send_json({"type": "tabs", "tabs": tabs, "active_tab_id": session.get("active_tab_id")})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    finally:
        streaming = False
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass


# Cleanup on shutdown
async def cleanup_sessions():
    await session_manager.cleanup()
