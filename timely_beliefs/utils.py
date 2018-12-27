from datetime import datetime, timedelta


def datetime_x_days_ago_at_y_oclock(end: datetime, x: int, y: int) -> datetime:
    return end.replace(hour=y) - timedelta(days=x)


def timedelta_x_days_ago_at_y_oclock(end: datetime, x: int, y: int) -> timedelta:
    return end - (
        end.replace(hour=y, minute=0, second=0, microsecond=0) - timedelta(days=x)
    )


def create_beliefs_query():
    pass
