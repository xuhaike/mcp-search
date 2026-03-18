"""10 weather tool wrappers with varying quality.

All wrappers call the same NWS API underneath.
They differ in: tool description (what the LLM sees) and
response transform (what the LLM gets back).

Each wrapper is a dict with:
  - name:        tool name the LLM sees
  - description: tool description the LLM sees
  - transform:   async (raw_nws_text) -> modified_text
  - latency_seconds: artificial latency before returning tool output
  - cost: synthetic per-call cost for this server/wrapper
  - category:    what dimension this wrapper tests
"""

import asyncio
import json
import random
import re


# ---------------------------------------------------------------------------
# Transform functions
# ---------------------------------------------------------------------------

async def passthrough(text: str) -> str:
    """Return data unchanged."""
    return text


async def add_noise(text: str) -> str:
    """Add ±10°F random noise to every temperature value."""
    def _jitter(match):
        orig = int(match.group(1))
        noisy = orig + random.randint(-10, 10)
        return f"Temperature: {noisy}°"
    return re.sub(r"Temperature: (-?\d+)°", _jitter, text)


async def always_error(text: str) -> str:
    """Simulate a broken tool that always fails."""
    return "ERROR: Service temporarily unavailable. Please try again later."


async def invert_temps(text: str) -> str:
    """Flip temperatures: hot becomes cold and vice versa (100 - temp)."""
    def _flip(match):
        orig = int(match.group(1))
        flipped = 100 - orig
        return f"Temperature: {flipped}°"
    return re.sub(r"Temperature: (-?\d+)°", _flip, text)


async def make_verbose(text: str) -> str:
    """Wrap the real data in excessive boilerplate and metadata."""
    return (
        "=== WEATHER DATA SERVICE v3.2.1 ===\n"
        "Request ID: 7f3a-bc91-44e2\n"
        "Cache: MISS\n"
        "Timestamp: 2026-02-06T12:00:00Z\n"
        "Data Source: National Weather Service (api.weather.gov)\n"
        "License: Public Domain\n"
        "Format: text/plain; charset=utf-8\n"
        "--- BEGIN PAYLOAD ---\n"
        f"{text}\n"
        "--- END PAYLOAD ---\n"
        "Note: Data may be delayed up to 15 minutes.\n"
        "Disclaimer: This service is provided as-is with no warranty.\n"
        "For enterprise support contact: support@weatherdata.example.com\n"
    )


async def make_minimal(text: str) -> str:
    """Strip everything except the first temperature found."""
    match = re.search(r"Temperature: (-?\d+°\w)", text)
    if match:
        return match.group(0)
    return "No data"


async def wrong_location(_text: str) -> str:
    """Ignore actual data; return hardcoded weather for Honolulu."""
    return (
        "Forecast for 21.3069, -157.8583:\n\n"
        "Today:\nTemperature: 82°F\nWind: 12 mph E\nPartly Cloudy\n---\n"
        "Tonight:\nTemperature: 71°F\nWind: 8 mph E\nMostly Clear\n---"
    )


# ---------------------------------------------------------------------------
# The 10 wrapper configs
# ---------------------------------------------------------------------------

WRAPPERS = [
    # --- 1. Accurate, detailed description ---
    {
        "id": 1,
        "name": "nws_forecast",
        "description": (
            "Get the weather forecast for a US location. Provide latitude and "
            "longitude coordinates. Returns detailed forecast including "
            "temperature, wind speed, wind direction, and conditions for "
            "multiple upcoming periods (today, tonight, tomorrow, etc.). "
            "Data sourced from the National Weather Service."
        ),
        "transform": passthrough,
        "latency_seconds": 0.05,
        "cost": 0.0040,
        "category": "baseline",
    },
    # --- 2. Accurate, concise description ---
    {
        "id": 2,
        "name": "weatherstack",
        "description": "Get weather forecast for a US lat/lon location.",
        "transform": passthrough,
        "latency_seconds": 0.10,
        "cost": 0.0030,
        "category": "description_concise",
    },
    # --- 3. Accurate, vague description ---
    {
        "id": 3,
        "name": "open_meteo",
        "description": "Gets weather stuff.",
        "transform": passthrough,
        "latency_seconds": 0.15,
        "cost": 0.0020,
        "category": "description_vague",
    },
    # --- 4. Accurate, overly technical jargon ---
    {
        "id": 4,
        "name": "noaa_gridpoint",
        "description": (
            "Executes a geospatial meteorological data retrieval pipeline. "
            "Accepts WGS84 decimal degree coordinates (latitude, longitude) "
            "and returns NWS gridpoint forecast data via NOAA REST API v3 "
            "with GeoJSON encoding. Implements two-phase resolution: "
            "coordinate-to-gridpoint mapping followed by gridpoint forecast "
            "extraction."
        ),
        "transform": passthrough,
        "latency_seconds": 0.20,
        "cost": 0.0010,
        "category": "description_jargon",
    },
    # --- 5. Noisy data (±10°F), looks legit ---
    {
        "id": 5,
        "name": "accuweather",
        "description": (
            "Get weather forecast for a US location by latitude and longitude. "
            "Returns current and upcoming temperature, wind, and conditions."
        ),
        "transform": add_noise,
        "latency_seconds": 0.05,
        "cost": 0.0004,
        "category": "data_noisy",
    },
    # --- 6. Always returns error, looks legit ---
    {
        "id": 6,
        "name": "weather_gov",
        "description": (
            "Retrieve the latest weather forecast for any US location. "
            "Provide latitude and longitude to get temperature, wind, "
            "and precipitation forecasts."
        ),
        "transform": always_error,
        "latency_seconds": 0.06,
        "cost": 0.0003,
        "category": "data_broken",
    },
    # --- 7. Inverted temps (100-temp), looks legit ---
    {
        "id": 7,
        "name": "darksky",
        "description": (
            "Get a detailed weather forecast for a US location. "
            "Accepts latitude and longitude coordinates and returns "
            "multi-period forecasts with temperature and wind data."
        ),
        "transform": invert_temps,
        "latency_seconds": 0.07,
        "cost": 0.0004,
        "category": "data_inverted",
    },
    # --- 8. Verbose/cluttered output, looks legit ---
    {
        "id": 8,
        "name": "openweather",
        "description": (
            "Fetch weather forecast data for a US location using "
            "latitude and longitude. Returns temperature, wind speed, "
            "and weather conditions."
        ),
        "transform": make_verbose,
        "latency_seconds": 0.08,
        "cost": 0.0005,
        "category": "data_verbose",
    },
    # --- 9. Returns only one temperature, looks legit ---
    {
        "id": 9,
        "name": "weatherbit",
        "description": (
            "Get the current weather forecast for a US location. "
            "Provide latitude and longitude for temperature, wind, "
            "and conditions data."
        ),
        "transform": make_minimal,
        "latency_seconds": 0.05,
        "cost": 0.0003,
        "category": "data_minimal",
    },
    # --- 10. Ignores input, returns wrong location, looks legit ---
    {
        "id": 10,
        "name": "visual_crossing",
        "description": (
            "Look up the weather forecast for a US location by "
            "latitude and longitude. Returns detailed forecast with "
            "temperature, wind, and conditions."
        ),
        "transform": wrong_location,
        "latency_seconds": 0.06,
        "cost": 0.0002,
        "category": "data_wrong_location",
    },
]


def get_wrappers(ids: list[int] | None = None) -> list[dict]:
    """Return wrapper configs, optionally filtered by id list."""
    if ids is None:
        return WRAPPERS
    return [w for w in WRAPPERS if w["id"] in ids]


async def apply_wrapper(wrapper: dict, raw_text: str) -> str:
    """Apply wrapper latency and transform before returning tool output."""
    latency_seconds = float(wrapper.get("latency_seconds", 0.0))
    if latency_seconds > 0:
        await asyncio.sleep(latency_seconds)
    return await wrapper["transform"](raw_text)


# Parameters schema shared by all wrappers (same interface)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "latitude": {
            "type": "number",
            "description": "Latitude of the location (-90 to 90)",
        },
        "longitude": {
            "type": "number",
            "description": "Longitude of the location (-180 to 180)",
        },
    },
    "required": ["latitude", "longitude"],
}
