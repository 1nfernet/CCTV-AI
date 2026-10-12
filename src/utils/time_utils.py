"""
time_utils.py – Timestamp formatting and time-of-day classification.
"""

from datetime import datetime


def now_iso() -> str:
    """Return the current local time as an ISO-8601 string."""
    return datetime.now().isoformat(timespec="seconds")


def now_readable() -> str:
    """Return the current local time in a human-friendly format (e.g. ``7:12 PM``)."""
    return datetime.now().strftime("%-I:%M %p")


def format_timestamp(ts: str | datetime, fmt: str = "%-I:%M %p") -> str:
    """
    Format a timestamp string or datetime object.

    Parameters
    ----------
    ts : str or datetime
        ISO-8601 string or ``datetime`` instance.
    fmt : str
        ``strftime`` format string.

    Returns
    -------
    str
    """
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts.strftime(fmt)


def is_night(hour: int | None = None, start: int = 22, end: int = 6) -> bool:
    """
    Return ``True`` if the given hour (or the current hour) falls within
    the night window [start, end).

    Parameters
    ----------
    hour : int, optional
        Hour in 24-h format.  Defaults to the current hour.
    start : int
        Night-start hour (default 22 = 10 PM).
    end : int
        Night-end hour (default 6 = 6 AM).
    """
    if hour is None:
        hour = datetime.now().hour
    if start > end:                             # wraps midnight
        return hour >= start or hour < end
    return start <= hour < end


def time_of_day(hour: int | None = None) -> str:
    """
    Classify the current (or given) hour into a human-readable period.

    Returns one of: ``morning``, ``afternoon``, ``evening``, ``night``.
    """
    if hour is None:
        hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def seconds_since(ts: str | datetime) -> float:
    """Return the number of seconds elapsed since *ts*."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return (datetime.now() - ts).total_seconds()
