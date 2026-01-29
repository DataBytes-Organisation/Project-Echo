from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.Components.API.app.routers import (
    audio_upload_router,
    species_predictor,
    auth_router,
    hmi,
    engine,
    sim,
    two_factor,
    public,
    iot,
    detections,
    admin
)

# ✅ Create FastAPI app with metadata
app = FastAPI(
    title="Project Echo API",
    description="IoT-based system for audio species detection and ecosystem monitoring",
    version="1.0.0"
)

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Routers
app.include_router(audio_upload_router.router, prefix="/api", tags=["audio"])
app.include_router(species_predictor.router, prefix="/predict", tags=["predict"])
app.include_router(detections.router, prefix="/ws", tags=["detections"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(hmi.router, prefix="/hmi", tags=["hmi"])
app.include_router(engine.router, prefix="/engine", tags=["engine"])
app.include_router(sim.router, prefix="/sim", tags=["sim"])
app.include_router(two_factor.router)
app.include_router(public.router, prefix="/public", tags=["public"])
app.include_router(iot.router, prefix="/iot", tags=["iot"])
app.include_router(auth_router.router, prefix="/api", tags=["auth"])

# ✅ Root endpoint
@app.get("/", response_description="API Root")
def root():
    return "Welcome to Project Echo API. Visit /docs for interactive documentation."