from datetime import datetime, timedelta
from inspect import getfullargspec, getmembers, isfunction
from typing import Optional, Tuple, Union

from isodate import (
    ISO8601Error,
    datetime_isoformat,
    duration_isoformat,
    parse_datetime,
    parse_duration,
)

from timely_beliefs.sensors.func_store import knowledge_horizons
from timely_beliefs.sensors.func_store.utils import (  # noqa F401
    datetime_x_days_ago_at_y_oclock,
)  # third parties may have historically imported datetime_x_days_ago_at_y_oclock from here


def jsonify_time_dict(d: dict) -> dict:
    """Convert datetime and timedelta values in the dict to iso string format"""
    d2 = {}
    for k, v in d.items():
        if isinstance(v, datetime):
            d2[k] = datetime_isoformat(v)
        elif isinstance(v, timedelta):
            d2[k] = duration_isoformat(v)
        else:
            d2[k] = v
    return d2


def unjsonify_time_dict(d: dict) -> dict:
    """Convert datetime and timedelta values in the dict to iso string format"""
    d2 = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                d2[k] = parse_duration(v)
            except ISO8601Error:
                try:
                    d2[k] = parse_datetime(v)
                except ISO8601Error:
                    d2[k] = v
        else:
            d2[k] = v
    return d2


def get_func_store() -> dict:
    """Returns a dictionary with Callable objects supported in our function store,
    indexed by their function names."""
    return {o[0]: o[1] for o in getmembers(knowledge_horizons) if isfunction(o[1])}


def eval_verified_knowledge_horizon_fnc(
    requested_fnc_name: str,
    par: dict,
    event_start: Optional[datetime] = None,
    event_resolution: timedelta = None,
    get_bounds: bool = False,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Evaluate knowledge horizon function to return a knowledge horizon.
    Only function names that represent Callable objects in our function store can be evaluated.
    If get_bounds is True, a tuple is returned with bounds on the possible return value.
    """
    for verified_fnc_name, verified_fnc in get_func_store().items():
        if verified_fnc_name == requested_fnc_name:
            if {"event_start", "event_resolution"} < set(
                getfullargspec(verified_fnc).args
            ):
                # Knowledge horizons are anchored to event_end = event_start + event_resolution
                return verified_fnc(
                    event_start,
                    event_resolution,
                    **(unjsonify_time_dict(par)),
                    get_bounds=get_bounds
                )
            elif "event_start" in getfullargspec(verified_fnc).args:
                # Knowledge horizons are anchored to event_start
                return verified_fnc(
                    event_start, **(unjsonify_time_dict(par)), get_bounds=get_bounds
                )
            elif "event_resolution" in getfullargspec(verified_fnc).args:
                # Knowledge horizons are anchored to event_start
                return verified_fnc(
                    event_resolution,
                    **(unjsonify_time_dict(par)),
                    get_bounds=get_bounds
                )
    raise Exception(
        "knowledge_horizon_fnc %s cannot be executed safely. Please register the function in the func_store."
        % requested_fnc_name
    )
