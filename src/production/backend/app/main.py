import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import ConnectionFailure, ExecutionTimeout

from app.cache import close_client as close_cache_client
from app.config import settings
from app.database import client, Userclient
from app.errors import (
    StandardizeErrorResponseMiddleware,
    detection_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.exceptions import DetectionError
from app.logging_config import configure_logging
from app.metrics import PrometheusMiddleware, metrics_router
from app.middleware.correlation_id import add_correlation_id
from app.queue import redis_conn
from app.routers import (
    admin_budget,
    admin_services,
    audio_upload_router,
    auth_router,
    detections,
    engine,
    health,
    hmi,
    insights,
    iot,
    live,
    payments,
    projects,
    public,
    sensors,
    sim,
    similar_detections,
    species_predictor,
    two_factor,
)
from app.services.mqtt_client import (
    get_connection_state,
    get_latest_events,
    start_mqtt_client,
    stop_mqtt_client,
)

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info(f"API Server listening on port {settings.api_port}")
    try:
        start_mqtt_client()
    except Exception as e:
        logger.warning("Could not automatically start MQTT client: %s", e)
    yield
    # Shutdown actions
    logger.info("Backend shutting down... Gracefully closing all owned client connections.")

    # 1. Stop the MQTT client: disconnect from the broker and join its loop thread
    try:
        stop_mqtt_client()
    except Exception as e:
        logger.error("Error stopping MQTT client: %s", e)

    # 2. Close Primary EchoNet MongoDB Client
    try:
        client.close()
        logger.info("Primary MongoDB client closed cleanly.")
    except Exception as e:
        logger.error("Error closing primary MongoDB client: %s", e)

    # 3. Close User MongoDB Client
    try:
        Userclient.close()
        logger.info("User MongoDB client closed cleanly.")
    except Exception as e:
        logger.error("Error closing user MongoDB client: %s", e)

    # 4. Close Redis Connection
    try:
        redis_conn.close()
        logger.info("Redis queue connection closed cleanly.")
    except Exception as e:
        logger.error("Error closing Redis connection: %s", e)

    # 5. Close the insights Redis cache connection (DB 2)
    try:
        close_cache_client()
        logger.info("Redis cache connection closed cleanly.")
    except Exception as e:
        logger.error("Error closing Redis cache connection: %s", e)


app = FastAPI(
    lifespan=lifespan,
    title="Project Echo API",
    description="""
    Project Echo is an IoT-based system designed to record and analyze audio data for species identification and ecosystem monitoring.

    This API provides endpoints to:
    - Upload audio files
    - Simulate audio responses
    - Interface with HMI and audio engine modules
    """,
    version="1.0.0",
)


# Exception Handlers
async def database_unavailable_handler(request: Request, exc: Exception):
    logger.warning(
        "Database unavailable for %s %s (%s)",
        request.method,
        request.url.path,
        exc.__class__.__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Database Unavailable",
            "message": "The database is temporarily unavailable. Please try again later.",
        },
    )


app.add_exception_handler(ConnectionFailure, database_unavailable_handler)
app.add_exception_handler(ExecutionTimeout, database_unavailable_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(DetectionError, detection_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Middlewares
app.add_middleware(StandardizeErrorResponseMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus HTTP request metrics middleware
app.add_middleware(PrometheusMiddleware)


# Routers
app.include_router(health.router)
app.include_router(projects.router)
app.include_router(audio_upload_router.router, tags=["audio"], prefix="/api")
app.include_router(hmi.router, tags=["hmi"], prefix="/hmi")
app.include_router(engine.router, tags=["engine"], prefix="/engine")
app.include_router(sim.router, tags=["sim"], prefix="/sim")
app.include_router(two_factor.router)
app.include_router(admin_budget.router, tags=["admin"], prefix="/api")
app.include_router(admin_services.router, tags=["admin"], prefix="/api")
app.include_router(public.router, tags=["public"], prefix="/public")
app.include_router(iot.router, tags=["iot"], prefix="/iot")
app.include_router(sensors.router, tags=["sensors"], prefix="/sensors")
app.include_router(payments.router)
app.include_router(live.router, tags=["live"])
app.include_router(species_predictor.router, tags=["predict"])
app.include_router(insights.router, tags=["insights"])
app.include_router(auth_router.router, tags=["auth"], prefix="/api")
app.include_router(detections.router)
app.include_router(similar_detections.router)

# Prometheus /metrics endpoint
app.include_router(metrics_router)


# Root Endpoint
@app.get("/", response_description="API Root")
def show_home():
    return "Welcome to Project Echo API. Visit /docs for interactive documentation."


# MQTT Endpoints (FR-A2)
@app.get("/mqtt/connection-state", tags=["mqtt"])
def mqtt_connection_state():
    return {"state": get_connection_state()}


@app.get("/mqtt/latest-events", tags=["mqtt"])
def mqtt_latest_events():
    from app.routers.hmi import show_latest_events

    events = [
        {"eventType": "vocalization", **event}
        for event in show_latest_events(limit=20)
    ]
    events += [
        event
        for event in get_latest_events()
        if event.get("eventType") != "vocalization"
    ]
    return {"events": events}


# Correlate every response and application log entry (C9.2). This middleware
# is registered last so it is the outermost user middleware and can observe
# responses generated by the other middleware.
add_correlation_id(app)


# OpenAPI Exporters
@app.get("/openapi-export", include_in_schema=False)
async def get_openapi_spec():
    return app.openapi()


@app.get("/spec/summary", tags=["debug"], include_in_schema=False)
async def get_spec_summary():
    spec = app.openapi()
    return {
        "title": spec.get("info", {}).get("title"),
        "version": spec.get("info", {}).get("version"),
        "number_of_paths": len(spec.get("paths", {})),
        "tags": [tag.get("name") for tag in spec.get("tags", []) if "name" in tag],
    }


def export_openapi_to_file():
    output_dir = "backend"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "project-echo-openapi.json")
    try:
        with open(output_path, "w") as f:
            json.dump(app.openapi(), f, indent=2)
        logger.info(f"OpenAPI spec exported to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to export OpenAPI spec: {e}")


export_openapi_to_file()
