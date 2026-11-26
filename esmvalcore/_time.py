"""Parse recipe time ranges consistently for selection and preprocessing."""

from __future__ import annotations

import datetime as dt

import isodate


def _parse_start_date(
    date: str,
) -> dt.datetime | dt.timedelta | isodate.duration.Duration:
    """Parse start of the input `timerange` tag given in ISO 8601 format.

    Returns a datetime.datetime object.

    Raises an ISO8601 parser error if data can not be parsed.
    """
    if date.startswith("P"):
        start_date = isodate.parse_duration(date)
    elif "T" in date:
        start_date = isodate.parse_datetime(date)
    else:
        start_date = isodate.parse_date(date)
        start_date = dt.datetime.combine(start_date, dt.time.min)

    return start_date


def _parse_end_date(
    date: str,
) -> dt.datetime | dt.timedelta | isodate.duration.Duration:
    """Parse end of the input `timerange` given in ISO 8601 format.

    Returns a datetime.datetime object.

    Raises an ISO8601 parser error if data can not be parsed.
    """
    if date.startswith("P"):
        end_date = isodate.parse_duration(date)
    elif len(date) == 4:
        end_date = dt.datetime(int(date) + 1, 1, 1, 0, 0, 0)
    elif len(date) == 6:
        month = int(date[4:]) + 1
        year = int(date[0:4])
        if month == 13:
            month = 1
            year += 1
        end_date = dt.datetime(year, month, 1, 0, 0, 0)
    else:
        if "T" in date:
            end_date = isodate.parse_datetime(date)
        else:
            end_date = isodate.parse_date(date)
            end_date = dt.datetime.combine(
                end_date,
                dt.time.min,
            )
        end_date += dt.timedelta(seconds=1)
    return end_date


def _duration_to_date(
    duration: dt.timedelta | isodate.duration.Duration,
    reference: dt.datetime,
    sign: int,
) -> dt.datetime:
    """Add or subtract a duration period to a reference datetime."""
    return reference + sign * duration


def _parse_timerange(timerange: str) -> tuple[dt.datetime, dt.datetime]:
    """Resolve a recipe time range to inclusive start and exclusive end."""
    start_date = _parse_start_date(timerange.split("/", maxsplit=1)[0])
    end_date = _parse_end_date(timerange.split("/")[1])

    if not isinstance(start_date, dt.datetime):
        if not isinstance(end_date, dt.datetime):
            msg = "A time range must contain at least one date."
            raise TypeError(msg)
        duration = start_date
        start_date = _duration_to_date(duration, end_date, sign=-1)
        if isinstance(duration, dt.timedelta):
            start_date -= dt.timedelta(seconds=1)
    elif not isinstance(end_date, dt.datetime):
        duration = end_date
        end_date = _duration_to_date(duration, start_date, sign=1)
        if isinstance(duration, dt.timedelta):
            end_date += dt.timedelta(seconds=1)

    return start_date, end_date
