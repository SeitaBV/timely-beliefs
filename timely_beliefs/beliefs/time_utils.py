from __future__ import annotations

from datetime import timedelta

from isodate import parse_duration
import pandas as pd


def to_max_timedelta(duration: str | pd.DateOffset | timedelta) -> pd.Timedelta:
    """Determine the maximum pd.Timedelta for a given ISO duration string or Pandas DateOffset object.

    - Converts years to 366 days and months to 31 days.
    - Does not convert days to 25 hours.
    """
    if isinstance(duration, timedelta):
        return pd.Timedelta(duration)
    if isinstance(duration, pd.DateOffset):
        duration = offset_to_iso_duration(duration)

    offset_args = _iso_duration_to_offset_args(duration)
    years = offset_args.get("years", 0)
    months = offset_args.get("months", 0)
    days = offset_args.get("days", 0)
    if years:
        days += 366 * years
        offset_args["years"] = 0
    if months:
        days += 31 * months
        offset_args["months"] = 0
    offset_args["days"] = days

    return pd.Timedelta(**offset_args)


def _iso_duration_to_offset_args(iso_duration: str) -> dict[str, int]:
    # Validate ISO duration string before commencing our own parsing
    parse_duration(iso_duration)

    # Initialize offset components
    offset_args = {}

    # Parsing ISO duration string
    pos = 0
    encountered_time_designator = False
    while pos < len(iso_duration):
        num = ""
        while pos < len(iso_duration) and iso_duration[pos].isdigit():
            num += iso_duration[pos]
            pos += 1

        if pos >= len(iso_duration):
            break

        if not encountered_time_designator and iso_duration[pos] == "Y":
            offset_args["years"] = int(num)
        elif not encountered_time_designator and iso_duration[pos] == "M":
            offset_args["months"] = int(num)
        elif not encountered_time_designator and iso_duration[pos] == "W":
            offset_args["weeks"] = int(num)
        elif not encountered_time_designator and iso_duration[pos] == "D":
            offset_args["days"] = int(num)
        elif iso_duration[pos] == "T":
            encountered_time_designator = True
        elif encountered_time_designator and iso_duration[pos] == "H":
            offset_args["hours"] = int(num)
        elif encountered_time_designator and iso_duration[pos] == "M":
            offset_args["minutes"] = int(num)
        elif encountered_time_designator and iso_duration[pos] == "S":
            offset_args["seconds"] = int(num)
        pos += 1

    return offset_args


def iso_duration_to_offset(iso_duration: str) -> pd.DateOffset:
    """
    Convert an ISO duration string to a Pandas DateOffset object.

    :param iso_duration:    ISO duration string to convert.
    :return:                Pandas DateOffset object representing the duration.
    """
    offset_args = _iso_duration_to_offset_args(iso_duration)

    # Construct DateOffset without zero-valued components
    return pd.DateOffset(**{k: v for k, v in offset_args.items() if v})


def _get_nominal_period_from_offset(
    offset: pd.DateOffset, name: str, designator: str
) -> str:
    try:
        n_periods = getattr(offset, name)
        if n_periods != 0:
            return str(n_periods) + designator
    except AttributeError:
        pass
    return ""


def _offset_contains_time(offset: pd.DateOffset) -> bool:
    """Also returns False if the offset contains only zero-valued time components."""
    n_hours = False
    n_minutes = False
    n_seconds = False
    try:
        n_hours = offset.hours
    except AttributeError:
        pass
    try:
        n_minutes = offset.minutes
    except AttributeError:
        pass
    try:
        n_seconds = offset.seconds
    except AttributeError:
        pass
    return bool(n_hours * n_minutes * n_seconds)


def offset_to_iso_duration(offset: pd.DateOffset) -> str:
    """
    Convert a Pandas DateOffset to an ISO duration string.

    Parameters:
        offset (DateOffset): Pandas DateOffset object to convert.

    Returns:
        str: ISO duration string representing the duration of the offset.
    """
    iso_duration = "P"
    iso_duration += _get_nominal_period_from_offset(offset, "years", "Y")
    iso_duration += _get_nominal_period_from_offset(offset, "months", "M")
    iso_duration += _get_nominal_period_from_offset(offset, "weeks", "W")
    iso_duration += _get_nominal_period_from_offset(offset, "days", "D")

    # check for hours/minutes/seconds
    if _offset_contains_time(offset):
        iso_duration += "T"
        iso_duration += _get_nominal_period_from_offset(offset, "hours", "H")
        iso_duration += _get_nominal_period_from_offset(offset, "minutes", "M")
        iso_duration += _get_nominal_period_from_offset(offset, "seconds", "S")

    if iso_duration == "P":
        iso_duration = "PT0H"

    return iso_duration
