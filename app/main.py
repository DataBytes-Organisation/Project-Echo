# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import asyncio, os, time
from .model import DummyModel

app = FastAPI(title="EchoNet Model API", version="0.1.0")

# 控制并发（避免推理过载）：默认同时处理 2 个请求
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
_sema = asyncio.Semaphore(MAX_CONCURRENCY)

_model = DummyModel()

class PredictResponse(BaseModel):
    label: str
    latency_ms: int

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metadata")
async def metadata():
    return {
        "model_name": _model.name,
        "version": "0.1.0",
        "max_concurrency": MAX_CONCURRENCY,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio/* type.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    start = time.perf_counter()
    async with _sema:
        loop = asyncio.get_event_loop()
        # 推理放在线程池，避免阻塞事件循环
        label = await loop.run_in_executor(None, _model.predict, data)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {"label": label, "latency_ms": latency_ms}
