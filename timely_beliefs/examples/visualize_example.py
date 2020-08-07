import pandas as pd

from timely_beliefs.examples import example_df as df

pd.set_option("display.max_columns", None)

chart = df.plot(reference_source=df.lineage.sources[0], interpolate=False)

chart.save("chart.json")
# chart.save("chart.png", scale_factor=1.5)  # See https://stackoverflow.com/questions/40555930/selenium-chromedriver-executable-needs-to-be-in-path
chart.serve(open_browser=False)
