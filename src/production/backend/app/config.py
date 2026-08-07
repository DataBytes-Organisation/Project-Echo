"""Central configuration for the Project Echo backend.

Every setting the application needs is read here, in one place, from
environment variables. Nothing in this module opens a network connection or
imports anything from the rest of the application, which is what makes it
safe to import from a test.

Two rules are applied:

* Secrets are required. If a connection string is missing, the application
  stops with a message naming the variable, rather than falling back to a
  value written into the source code.
* Non-secrets may have a default, because a wrong address is obvious and
  harmless, while a leaked password is neither.

Related task: B4.2 hardcoded URL and port audit, C1.5 environment variable
fallback tests.
"""

import os

# Connection string prefixes accepted by pymongo.
_MONGO_SCHEMES = ("mongodb://", "mongodb+srv://")


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or its value is invalid."""


def _read(name):
    """Return the raw value of ``name``, or None if unset or blank.

    An empty or whitespace-only variable is treated as unset. This matters
    because ``docker run -e MONGODB_URI=`` sets the variable to an empty
    string rather than leaving it out, and an empty connection string is not
    a usable value.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_required(name):
    """Return the value of ``name``, or raise ConfigError if it is not set."""
    value = _read(name)
    if value is None:
        raise ConfigError(
            "Required environment variable '{}' is not set. "
            "Set it in your .env file or container environment.".format(name)
        )
    return value


def get_optional(name, default):
    """Return the value of ``name``, or ``default`` if it is not set."""
    value = _read(name)
    return default if value is None else value


def get_list(name, default):
    """Return ``name`` parsed as a comma-separated list, or ``default``.

    Blank entries are discarded, so "a, ,b" yields ["a", "b"]. If every entry
    is blank the default is returned rather than an empty list, because an
    empty list of allowed origins would silently block all browser traffic.
    """
    value = _read(name)
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",")]
    items = [item for item in items if item]
    return items or list(default)


def _require_mongo_uri(name):
    """Return ``name`` as a MongoDB URI, rejecting anything malformed."""
    value = get_required(name)
    if not value.startswith(_MONGO_SCHEMES):
        raise ConfigError(
            "Environment variable '{}' does not look like a MongoDB "
            "connection string. Expected it to start with one of {}, got "
            "'{}'.".format(name, " or ".join(_MONGO_SCHEMES), value)
        )
    return value


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
# Each setting is a function rather than a module-level constant so that the
# environment is read at call time. A constant would freeze whatever the
# environment happened to be when the module was first imported, which makes
# the behaviour impossible to test and surprising in practice.


def mongodb_uri():
    """Connection string for the EchoNet database. Required, no default."""
    return _require_mongo_uri("MONGODB_URI")


def user_mongodb_uri():
    """Connection string for the UserSample database. Required, no default."""
    return _require_mongo_uri("USER_MONGODB_URI")


def api_base_url():
    """Base address the backend uses when calling its own API.

    Defaults to the local development address. Not a secret, so a default is
    acceptable; a wrong value fails loudly and immediately.
    """
    return get_optional("API_BASE_URL", "http://localhost:9000").rstrip("/")


def cors_allowed_origins():
    """Browser origins permitted to call this API.

    Defaults to the local HMI development server.
    """
    return get_list("CORS_ALLOWED_ORIGINS", ["http://localhost:8080"])
