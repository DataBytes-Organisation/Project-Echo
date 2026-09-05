import json
import logging
import sys
from datetime import datetime, timezone

from app.config import settings


# Format Python log records as single-line JSON objects.
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = (datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"))
        log_entry = {
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "timestamp": timestamp
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


# Configure application and Uvicorn logging to output JSON.
def configure_logging() -> None:
    log_level = settings.log_level.upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    # Forward Uvicorn logs through the JSON formatter
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True