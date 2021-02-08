import os
from datetime import timedelta

import timely_beliefs as tb
from timely_beliefs.examples.beliefs_data_frames import sixteen_probabilistic_beliefs

example_df = sixteen_probabilistic_beliefs()
temperature_df = tb.read_csv(
    os.path.dirname(os.path.abspath(__file__)) + "/temperature.csv",
    sensor=tb.Sensor("Thermometer A", unit="°C", event_resolution=timedelta(hours=1)),
    source=tb.BeliefSource("Source X"),
)
