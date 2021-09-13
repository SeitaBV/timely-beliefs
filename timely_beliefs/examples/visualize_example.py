import pandas as pd

from timely_beliefs.examples import get_example_df

pd.set_option("display.max_columns", None)

example_df = get_example_df()
chart = example_df.plot(
    reference_source=example_df.lineage.sources[0], interpolate=False
)

chart.save("chart.json")
# chart.save("chart.png", scale_factor=1.5)  # See https://stackoverflow.com/questions/40555930/selenium-chromedriver-executable-needs-to-be-in-path
chart.serve(open_browser=False)
