## app.config.py
# Central typed configuration for the Backend API.
# All Backend code should read configuration through `settings` rather than
# calling os.getenv()/hardcoding values directly.
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # --- Database (required — fail fast if missing) ---
    mongodb_uri: str = Field(...)
    user_mongodb_uri: str = Field(...)
    mongo_db_name: str = Field("EchoNet", env="MONGO_DB")

    # --- Redis (for C1/C2, not yet used by this backend) ---
    redis_host: str = "echo-redis"
    redis_port: int = 6379

    # --- MQTT / HiveMQ ---
    mqtt_host: str = "ts-mqtt-server-cont"
    mqtt_port: int = 1883
    mqtt_engine_topic: str = "projectecho/engine/2"
    # Placeholder for the A1 inbound ingestion bridge; confirm/rename with the
    # IoT/Engine topic contract before A1 relies on this value.
    mqtt_detection_topic: str = "projectecho/detections"

    # --- API self-reference ---
    internal_api_base_url: str = "http://ts-api-cont:9000"
    api_port: int = 9000

    # --- Timeouts / thresholds (consumed by C10/C11 later) ---
    request_timeout_seconds: float = 15.0
    slow_operation_ms: float = 500.0
    cache_ttl_seconds: int = 60

    # --- Auth (required — fail fast if missing) ---
    jwt_secret: str = Field(...)
    jwt_algorithm: str = "HS256"
    jwt_expiry_seconds: int = 86400
    otp_expiry_minutes: int = 5

    # --- Logging / CORS ---
    log_level: str = "INFO"
    cors_origins: List[str] = ["*"]

    # --- Twilio (optional — SMS/2FA degrades gracefully if unset) ---
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None

    # --- Mail (optional) ---
    mail_username: Optional[str] = None
    mail_password: Optional[str] = None
    mail_from: Optional[str] = None
    mail_server: str = "smtp.gmail.com"
    mail_port: int = 587
    mail_starttls: bool = True
    mail_use_ssl: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
