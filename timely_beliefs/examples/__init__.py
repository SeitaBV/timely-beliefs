import os
from datetime import timedelta

import timely_beliefs as tb
from timely_beliefs.examples.beliefs_data_frames import sixteen_probabilistic_beliefs


def get_example_df():
    return sixteen_probabilistic_beliefs()


def get_temperature_df():
    return tb.read_csv(
        os.path.join(get_examples_path(), "temperature.csv"),
        sensor=tb.Sensor(
            "Thermometer A", unit="°C", event_resolution=timedelta(hours=1)
        ),
        source=tb.BeliefSource("Source X"),
    )


def get_examples_path() -> str:
    return os.path.dirname(os.path.abspath(__file__))
