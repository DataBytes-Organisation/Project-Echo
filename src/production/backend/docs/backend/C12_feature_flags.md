# C12 Backend Feature-Flag Framework

## Purpose

C12 provides a reusable mechanism for enabling or disabling selected Sprint 2
Backend behaviour through the central C4 typed configuration model.

Feature flags must be defined in `app/config.py`. Backend modules should not
introduce standalone environment-variable reads or scattered boolean checks.

## Current Flags

| Environment Variable | Settings Field | Default | Controls |
|---|---|---:|---|
| `REALTIME_STREAMING_ENABLED` | `realtime_streaming_enabled` | `true` | Detection WebSocket streaming |
| `ANALYTICS_EXTENSIONS_ENABLED` | `analytics_extensions_enabled` | `true` | Sprint 2 analytics filtering extensions |

## Reusable Helpers

Feature state is evaluated through:

- `is_feature_enabled(flag_name)`
- `require_feature(flag_name, display_name)`

Unknown feature names fail explicitly rather than silently falling back to an
enabled or disabled state.

For HTTP functionality, `require_feature()` returns a predictable HTTP 404
response when the feature is disabled.

WebSocket functionality checks `is_feature_enabled()` before connecting the
client.

## Adding a New Feature Flag

1. Add a boolean field to `Settings` in `app/config.py`.

Example:

example_feature_enabled: bool = Field(
    True,
    env="EXAMPLE_FEATURE_ENABLED",
)

2. Add the environment variable to .env.example.

EXAMPLE_FEATURE_ENABLED=true

3. Add a named constant to app/feature_flags.py.
EXAMPLE_FEATURE_FLAG = "example_feature_enabled"

4. Use require_feature() for HTTP behaviour or
is_feature_enabled() where an HTTP exception is not appropriate.

5. Add tests demonstrating both enabled and disabled behaviour.

## Current Behaviour
### Realtime streaming

When REALTIME_STREAMING_ENABLED=false, /ws/detections rejects the
connection before authentication or connection registration.

When enabled, the existing authentication and streaming behaviour is preserved.

### Analytics extensions

When ANALYTICS_EXTENSIONS_ENABLED=false, requests using Sprint 2 analytics
filters are rejected predictably.

Existing baseline analytics behaviour remains available when no Sprint 2 filter
is requested.

## Validation

C12 automated tests cover:

default enabled state
environment-based disabling
enabled/disabled reusable helper behaviour
unknown feature names
predictable HTTP disabled response
analytics filtering enabled/disabled behaviour
realtime streaming enabled/disabled behaviour

C4 + C12 regression tests are also executed together.