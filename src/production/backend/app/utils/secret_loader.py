import os


class SecretConfigurationError(RuntimeError):
    """Raised when a required secret is missing or insecure."""
    pass


INSECURE_SECRET_VALUES = {
    "changeme",
    "change-me",
    "password",
    "secret",
    "default",
    "example",
}


def load_secret(name: str) -> str:
    value = os.getenv(name)

    if value is None:
        raise SecretConfigurationError(
            f"Required environment variable '{name}' is not configured."
        )

    value = value.strip()

    if not value:
        raise SecretConfigurationError(
            f"Required environment variable '{name}' cannot be empty."
        )

    if value.lower() in INSECURE_SECRET_VALUES:
        raise SecretConfigurationError(
            f"Environment variable '{name}' contains an insecure placeholder value."
        )

    return value