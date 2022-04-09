# isort: skip_file
# flake8: noqa
import sys

from timely_beliefs.sensors.classes import DBSensor, Sensor, SensorDBMixin  # isort:skip
from timely_beliefs.sources.classes import (  # isort:skip
    BeliefSource,
    BeliefSourceDBMixin,
    DBBeliefSource,
)
from timely_beliefs.beliefs.classes import (
    BeliefsDataFrame,
    BeliefsSeries,
    DBTimedBelief,
    TimedBelief,
    TimedBeliefDBMixin,
)
from timely_beliefs.beliefs.utils import load_time_series, read_csv
from timely_beliefs.examples import beliefs_data_frames


__version__ = "Unknown"

# Our way to get the version uses importlib.metadata (added in Python 3.8)
# and relies on setuptools_scm.
# TODO: When timely-beliefs stops supporting Python 3.6 and 3.7,
#       we can remove the pkg_resources option.
if sys.version_info[:2] >= (3, 8):
    from importlib_metadata import version, PackageNotFoundError

    try:
        __version__ = version("timely_beliefs")
    except PackageNotFoundError:
        # package is not installed
        pass
else:
    import pkg_resources

    try:
        __version__ = pkg_resources.get_distribution("timely_beliefs").version
    except pkg_resources.DistributionNotFound:
        # package is not installed
        pass
