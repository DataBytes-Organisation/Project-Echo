# app/routers/live.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio, json, uuid
from typing import Dict

router = APIRouter()

# Tiny in-memory broadcaster
class Notifier:
    def __init__(self):
        self._subs: Dict[str, asyncio.Queue[str]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self) -> str:
        q: asyncio.Queue[str] = asyncio.Queue()
        sid = uuid.uuid4().hex
        async with self._lock:
            self._subs[sid] = q
        return sid

    async def unsubscribe(self, sid: str):
        async with self._lock:
            self._subs.pop(sid, None)

    async def publish(self, event: str, data):
        msg = json.dumps({"event": event, "data": data})
        async with self._lock:
            for q in self._subs.values():
                try: q.put_nowait(msg)
                except: pass

notifier = Notifier()

@router.get("/live/sse")
async def sse():
    sid = await notifier.subscribe()
    async def stream():
        try:
            # initial hello
            yield "event: hello\ndata: {\"msg\":\"connected\"}\n\n"
            # stream messages
            while True:
                msg = await notifier._subs[sid].get()
                yield f"event: update\ndata: {msg}\n\n"
        finally:
            await notifier.unsubscribe(sid)
    return StreamingResponse(stream(), media_type="text/event-stream")
