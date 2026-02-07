"""Direct NWS (National Weather Service) API client.

Mirrors the output format of the Node.js weather-mcp-server
but without any MCP overhead — just plain async functions.
"""

import httpx

NWS_API_BASE = "https://api.weather.gov"
HEADERS = {
    "User-Agent": "weather-experiment/1.0",
    "Accept": "application/geo+json",
}


async def get_forecast(latitude: float, longitude: float) -> str:
    """Fetch weather forecast for a lat/lon coordinate (US only)."""
    async with httpx.AsyncClient(headers=HEADERS, timeout=15) as client:
        # Step 1: get grid point
        points_url = f"{NWS_API_BASE}/points/{latitude:.4f},{longitude:.4f}"
        resp = await client.get(points_url)
        resp.raise_for_status()
        forecast_url = resp.json()["properties"]["forecast"]

        # Step 2: get forecast
        resp = await client.get(forecast_url)
        resp.raise_for_status()
        periods = resp.json()["properties"]["periods"]

    lines = [f"Forecast for {latitude}, {longitude}:\n"]
    for p in periods:
        lines.append(
            f"{p.get('name', 'Unknown')}:\n"
            f"Temperature: {p.get('temperature', '?')}°{p.get('temperatureUnit', 'F')}\n"
            f"Wind: {p.get('windSpeed', '?')} {p.get('windDirection', '')}\n"
            f"{p.get('shortForecast', '')}\n"
            f"---"
        )
    return "\n".join(lines)


async def get_alerts(state: str) -> str:
    """Fetch active weather alerts for a US state (2-letter code)."""
    async with httpx.AsyncClient(headers=HEADERS, timeout=15) as client:
        url = f"{NWS_API_BASE}/alerts?area={state.upper()}"
        resp = await client.get(url)
        resp.raise_for_status()
        features = resp.json().get("features", [])

    if not features:
        return f"No active alerts for {state.upper()}"

    lines = [f"Active alerts for {state.upper()}:\n"]
    for f in features:
        props = f["properties"]
        lines.append(
            f"Event: {props.get('event', 'Unknown')}\n"
            f"Area: {props.get('areaDesc', 'Unknown')}\n"
            f"Severity: {props.get('severity', 'Unknown')}\n"
            f"Status: {props.get('status', 'Unknown')}\n"
            f"Headline: {props.get('headline', 'No headline')}\n"
            f"---"
        )
    return "\n".join(lines)


# Quick self-test
if __name__ == "__main__":
    import asyncio

    async def _test():
        print(await get_forecast(40.7128, -74.0060))
        print()
        print(await get_alerts("CA"))

    asyncio.run(_test())
