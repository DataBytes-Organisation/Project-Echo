"""Domain exceptions raised by the Project Echo backend.

Keeping these exceptions independent of FastAPI lets the service layer report
what went wrong without choosing an HTTP response.  The application-wide
exception handler turns them into the standard API error envelope.
"""


class DetectionError(Exception):
    """Base class for expected failures in the detection service."""

    status_code = 500
    error_type = "detection_error"


class DetectionRuleError(DetectionError):
    """A request violates a detection validation or update rule."""

    status_code = 400
    error_type = "detection_rule_error"


class DetectionNotFoundError(DetectionError):
    """The requested detection does not exist."""

    status_code = 404
    error_type = "detection_not_found"


class DetectionStorageError(DetectionError):
    """MongoDB could not complete a detection operation."""

    status_code = 503
    error_type = "detection_storage_error"
