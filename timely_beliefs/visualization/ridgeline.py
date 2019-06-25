from datetime import datetime, timedelta
from timely_beliefs.beliefs.utils import load_time_series
from scipy.special import erfinv
from bokeh.palettes import viridis
from bokeh.io import show
from bokeh.models import ColumnDataSource, FixedTicker, FuncTickFormatter, LinearAxis
from bokeh.plotting import figure
import time, pickle
import datetime
import pytz
import isodate
import pandas as pd
import timely_beliefs as tb
import numpy as np
import scipy.stats as stats

#code to read in the csv file:
def read_beliefs_from_csv(sensor, source, cp, event_resolution: timedelta, tz_hour_difference: float = 0) -> list:
    """
    Returns a timely_beliefs DataFrame read from a csv file

    @param sensor : beliefsensor
    @param source : BeliefSource
    @param cp : float, cummulative probability
    @param event_resolution : timedelta object, event resolution
    @param tz_hour_difference : float,time difference
    """
    sensor_descriptions = (("Temperature", "C"),)

    cols = [0, 1]  # Columns with datetime index and observed values
    horizons = list(range(0, 169, 1))
    cols.extend([h + 2 for h in horizons])
    n_horizons = 169
    n_events = 500
    beliefs = pd.read_csv("%s-%s-%s.csv" % (sensor.name.replace(' ', '_').lower(), source.name.replace(' ', '_').lower(), cp),
                          index_col=0, parse_dates=[0], date_parser=lambda col: pd.to_datetime(col, utc=True) - timedelta(hours=tz_hour_difference),
                          nrows=n_events, usecols=cols)
    beliefs = beliefs.resample(event_resolution).mean()
    assert beliefs.index.tzinfo == pytz.utc
    # Construct the BeliefsDataFrame by looping over the belief horizons
    blfs = load_time_series(beliefs.iloc[:, 0].head(n_events), sensor=sensor, source=source,
                            belief_horizon=timedelta(hours=0), cumulative_probability=0.5)  # load the observations (keep cp=0.5)
    for h in beliefs.iloc[:, 1 :n_horizons + 1] :
        try:
            blfs += load_time_series(beliefs[h].head(n_events), sensor=sensor, source=source,
                                     belief_horizon=(isodate.parse_duration(
                                     "PT%s" % h)) + event_resolution, cumulative_probability=cp)  # load the forecasts
        except isodate.isoerror.ISO8601Error:  # In case of old headers that don't yet follow the ISO 8601 standard
            blfs += load_time_series(beliefs[h].head(n_events), sensor=sensor, source=source,
                                     belief_horizon=(isodate.parse_duration(
                                     "%s" % h)) + event_resolution, cumulative_probability=cp)  # load the forecasts
    return blfs

def make_df(n_events = 100, n_horizons = 169, tz_hour_difference=-9, event_resolution=timedelta(hours=1)):
    """
    Returns DataFrame in which n events and n horizons are stored

    @param n_events: int,number of events in DataFrame
    @param n_horizons: int,number of horizons in DataFrame
    @param tz_hour_difference: float,time difference
    @param event_resolution: timedelta object,event resolution
    """
    sensor_descriptions = (("Temperature", "C"),)
    source = tb.BeliefSource(name="Random forest")
    sensors = (tb.Sensor(name=descr[0], unit=descr[1], event_resolution=event_resolution) for descr in sensor_descriptions)
    blfs=[]
    for sensor in sensors:
        blfs += read_beliefs_from_csv(sensor, source=source, cp=0.05, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
        blfs += read_beliefs_from_csv(sensor, source=source, cp=0.5, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
        blfs += read_beliefs_from_csv(sensor, source=source, cp=0.95, event_resolution=event_resolution, tz_hour_difference=tz_hour_difference)
        bdf = tb.BeliefsDataFrame(sensor=sensor, beliefs=blfs).sort_index()
    return bdf
#end of code to read in csv file


def create_cp_data(df, start, end, start_time, fixedviewpoint):
    """
    Returns 3 lists with values of 2 different cumulative probabilities.
    Solely 1 out of 3 is 0.5.

    @param df: BeliefsDataFrame, containing events, belief times, predictions and their cumulative probabilities of 0.05/0.5/0.95
    @param start: int,start of timedelta in hours
    @param end: int,end of timedelta in hours
    @param start_time: datetime object, start of event
    @param fixedviewpoint: BOOLEAN,if true plot based on future predictions
    """
    first_date = df.iloc[0].name[0]
    last_date = df.iloc[-1].name[0]
    # check if current time is in data frame
    if start_time < first_date or start_time > last_date :
        raise ValueError('Your start time is not in the dataframe')
    #get cp for fixed viewpoint or not
    if fixedviewpoint == True:
        df = df[df.index.get_level_values("event_start") >= start_time]
        bdf = df.fixed_viewpoint(start_time)
        end = len(bdf)
        cp0 = bdf.iloc[0].name[3]
        cp1 = bdf.iloc[1].name[3]
        cp2 = bdf.iloc[2].name[3]
    else:
        bdf = df.belief_history(event_start=start_time, belief_horizon_window=(timedelta(hours=start), timedelta(hours=end)))
        cp0 = bdf.iloc[0].name[2]
        cp1 = bdf.iloc[1].name[2]
        cp2 = bdf.iloc[2].name[2]
    #list of cps
    cp_list = [cp0, cp1, cp2]
    list_0 = []
    list_1 = []
    list_2 = []
    #make list for each cp has to include 0.5
    if 0.5 in cp_list:
        i = 0
        index = cp_list.index(0.5)
        for _, value in bdf.iterrows():
            i = i%3
            if i == 0:
                list_0  += [value[0]]
            elif i == 1:
                list_1 += [value[0]]
            elif i == 2:
                list_2 += [value[0]]
            i += 1
        cp_list.remove(0.5)
        lists = [list_0, list_1, list_2]
        mean_list = lists.pop(index)
        return (cp_list, mean_list, lists[0], lists[1])
    raise ValueError("No mean cp value")


def ridgeline_plot(start_time, df, start=0, end=168, fixedviewpoint = False):
    """
    Creates ridgeline plot by selecting a belief history about a specific event

    @param start_time : datetime string of selected event
    @param df : timely_beliefs DataFrame
    @param start : start of hours before event time
    @param end : end of hours before event time
    @param fixedviewpoint : if true create fixed viewpoint plot
    """
    #set params
    if fixedviewpoint == True:
        start = 0
    #out of bounds checks
    if end < 0 or end > 168:
        raise ValueError("End of the forecast horizon must be between 0 and 168 hours.")
    if start < 0 or start > end:
        raise ValueError("Start of the forecast horizon must be between 0 and 168 hours.")
    #to include last observation
    end += 1
    #get cps
    cp_list, pred_temp_05, pred_temp_0, pred_temp_2 = create_cp_data(df,start,end,start_time,fixedviewpoint)
    #make means and std
    mean = np.array([float(i) for i in pred_temp_05])
    std1 = np.array([(float(pred_temp_0[i])-float(pred_temp_05[i]))/(np.sqrt(2)*erfinv((2*cp_list[0])-1)) for i in range(len(pred_temp_05))])
    std2 = np.array([(float(pred_temp_2[i])-float(pred_temp_05[i]))/(np.sqrt(2)*erfinv((2*cp_list[1])-1)) for i in range(len(pred_temp_05))])
    std = (std1+std2)/2
    #plot everything
    show_plot(mean, std, start, end, fixedviewpoint)


def show_plot(mean, std, start, end, fixedviewpoint=False):
    """
    Creates and shows ridgeline plot
    @param mean: list of float, mean values
    @param std: list of float, standard deviation values
    @param start: int,start hours before event-time
    @param end: int,end hours before event-time
    @param fixedviewpoint : BOOLEAN, if true create fixed viewpoint plot
    """
    #amount of lines to draw
    nr_lines = len(mean)
    x = np.linspace(-10, 30, 500)
    frame = pd.DataFrame()
    #generate points for each line
    for i in range(nr_lines):
        frame["{}".format(i)] = stats.norm.pdf(x, mean[i], std[i])
    #set color pallete to viridis
    pallete = viridis(nr_lines)
    #set list reversed or not depending on viewpoint
    if fixedviewpoint:
        cats = list(frame.keys())
    else:
        cats = list(reversed(frame.keys()))
    source = ColumnDataSource(data=dict(x=x))
    #creat figure
    p = figure(y_range=cats, plot_width=900, x_range=(-5, 30), toolbar_location=None)
    #make a ridge line in the figure
    for i, cat in enumerate(reversed(cats)):
        y = ridge(cat, frame[cat], 50)
        source.add(y, cat)
        p.patch('x', cat, alpha=0.6, color=pallete[i], line_color="black", source=source)
    #added y axis with the right ticks etc depending on fixedviewpoint
    if fixedviewpoint:
        p.yaxis.axis_label = 'Number of hours after event_start'
        y_ticks = list(np.arange(0, nr_lines, 5))
        yaxis = LinearAxis(ticker=y_ticks)
        y_labels = list((np.arange(0, nr_lines, 5)))
    else:
        p.yaxis.axis_label = 'Number of hours before event_start'
        y_ticks = list(np.arange(end, 0, -5))
        yaxis = LinearAxis(ticker=y_ticks)
        y_labels = list(np.arange(start, end, 5))
    #map ticks on to a dict
    mapping_dict = {y_ticks[i]: str(y_labels[i]) for i in range(len(y_labels))}
    for i in range(nr_lines):
        if i not in mapping_dict:
            mapping_dict[i]=" "
    mapping_code = "var mapping = {};\n    return mapping[tick];\n    ".format(mapping_dict)
    #format ticks
    p.yaxis.formatter = FuncTickFormatter(code=mapping_code)
    #set plot atributes
    p.outline_line_color = None
    p.background_fill_color = "#ffffff"
    p.xaxis.ticker = FixedTicker(ticks=list(range(-20, 101, 10)))
    p.xaxis.axis_label = 'Temperature (Celcius)'
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#000000"
    p.xgrid.ticker = p.xaxis[0].ticker
    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None
    #add padding in y derection
    p.y_range.range_padding = 0.2 / (nr_lines / 168)
    show(p)
    
#makes a list for the rich line fucntion as input
def ridge(category, data, scale=100):
    return list(zip([category] * len(data), scale * data))

#tests
df = make_df()
ridgeline_plot(datetime.datetime(2015, 3, 1, 9, 0, tzinfo=pytz.utc), df, end=150, fixedviewpoint=False)
ridgeline_plot(datetime.datetime(2015, 3, 1, 9, 0, tzinfo=pytz.utc), df, fixedviewpoint=True)
