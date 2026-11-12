"""Exceptions surfaced to Home Assistant callers."""

from __future__ import annotations

from homeassistant.exceptions import HomeAssistantError


class BedNotConnectedError(HomeAssistantError, ConnectionError):
    """The bed could not be reached for a command.

    A ConnectionError, so internal handlers keep working, and a
    HomeAssistantError, so service callers (REST, websocket, the UI) receive
    the readable reason instead of an anonymous internal server error.
    """
