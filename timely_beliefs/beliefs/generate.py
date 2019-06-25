import datetime
import pytz
from datetime import timedelta
import numpy as np
import timely_beliefs as tb
from timely_beliefs.beliefs.utils import load_time_series
import pandas as pd
from sklearn.linear_model import LinearRegression

#for plotting
from bokeh.palettes import viridis
from bokeh.io import show
from bokeh.models import ColumnDataSource, FixedTicker, FuncTickFormatter, LinearAxis
from bokeh.plotting import figure
import  scipy.stats as stats

#makes a list for the rich line function as input
def ridge(category, data, scale=100):
    return list(zip([category] * len(data), scale * data))

#reading csv as dataframe code:
def read_beliefs_from_csv(sensor,csv_in, source, event_resolution: timedelta = None, tz_hour_difference: float = 0, n_events: int = None) -> list:
    beliefs = pd.read_csv(csv_in,
                          index_col=0, parse_dates=[0],
                          date_parser=lambda col : pd.to_datetime(col, utc=True) - timedelta(hours=tz_hour_difference),
                          nrows=n_events, usecols=["datetime", sensor.name.replace(' ', '_').lower()])
    if event_resolution is not None:
        beliefs = beliefs.resample(event_resolution).mean()
    assert beliefs.index.tzinfo == pytz.utc
    # Construct the BeliefsDataFrame by looping over the belief horizons
    blfs = load_time_series(beliefs[sensor.name.replace(' ', '_').lower()], sensor=sensor, source=source,
                            belief_horizon=timedelta(hours=0), cumulative_probability=0.5)  # load the observations (keep cp=0.5)
    return blfs

def csv_as_belief(csv_in,tz_hour_difference,n_events = None):
    sensor_descriptions = (
        ("Temperature", "°C"),
    )
    # Create source and sensors
    source_a = tb.BeliefSource(name="KNMI")
    sensors = (tb.Sensor(name=descr[0], unit=descr[1], event_resolution=timedelta(minutes=15)) for descr in sensor_descriptions)
    # Create BeliefsDataFrame
    for sensor in sensors:
        blfs = read_beliefs_from_csv(sensor,csv_in, source=source_a, tz_hour_difference=tz_hour_difference, n_events=n_events)
        df = tb.BeliefsDataFrame(sensor=sensor, beliefs=blfs).sort_index()
    return df
#end of reading csv as dataframe code:

def generator(df, beliefSeries, model):
    """
    Generates a new forecast at a given time using a model
    @param df : BeliefsDataframe containing all necessary data
    @param beliefSeries : BeliefSeries object
    @param model : model to use to generate new data
    """
    # check if model is linear_regression
    if isinstance(model, LinearRegression):
        X = np.array([[1], [2], [3], [4]]) # placeholder, takes one point only
        model.fit(X,X)
        return model.predict(np.array([[beliefSeries[0]]]))[0][0]
    # error message
    raise NotImplementedError("This model functionality has not yet been implemented/ Did you pass the correct model?")

#gets a beliefSeries from a certain BeliefsDataFrames at a certain time.
#using a datetime object
def get_beliefsSeries_from_event_start(df,datetime_object,current_time,value):
    return df.loc[datetime_object.strftime("%m/%d/%Y, %H:%M:%S"),value]

#gets a beliefSeries from a certain BeliefsDataFrames at a certain time.
#using a datetime str
def get_beliefsSeries_from_event_start_str(datetime_str,current_time):
    return df.loc[(datetime_str,current_time),'event_value']

def main(df, current_time, start_time, last_start_time=None, model=LinearRegression()):
    """
    Accepts a Beliefs Dataframe df and returns forecasts from start_time to last_start_time in timely beliefs rows
    @param df: Beliefs Dataframe
    @param current_time : datetime object, generate a forecast from this point
    @param start_time: datetime object,
    @param last_start_time: datetime object
    @param model : model to use to generate new data
    """
    if last_start_time == None:
        last_start_time = start_time
    first_date = df.iloc[0].name[0]
    last_date = df.iloc[-1].name[0]
    # check if current time is in data frame
    if current_time < first_date or current_time > last_date :
        raise ValueError('Your current_time is not in the dataframe')
    # get beliefseries from all the times
    current = get_beliefsSeries_from_event_start(df,current_time,current_time, 'event_value')
    start = get_beliefsSeries_from_event_start(df,start_time,current_time, 'event_value')
    last_start = get_beliefsSeries_from_event_start(df,last_start_time,current_time, 'event_value')
    # create list of beliefSeries
    beliefSeries_list = [start.copy()]
    blfs_list = []
    temp_time = start_time
    i = 0
    # loop over given time slot
    while temp_time <= last_start_time:
        if temp_time > last_date:
            i += 1
            blfs_list +=  [tb.TimedBelief(
                source= tb.BeliefSource(name='test'+ str(i)),
                sensor= df.sensor,
                value= generator(df,current,model),
                belief_time= current_time,
                event_start= temp_time,
                cumulative_probability= 0.5,
            )]
        else:
            beliefSeries_list += [get_beliefsSeries_from_event_start(df,temp_time,current_time,'event_value').copy()]
        temp_time += df.sensor.event_resolution
    df_1 = tb.BeliefsDataFrame(sensor=df.sensor, beliefs=blfs_list)
    # loops over all time steps
    for beliefSeries in beliefSeries_list:
        if beliefSeries.empty == False:
            beliefSeries[0] = generator(df, current, model)
    temp = beliefSeries_list[0].to_frame(name='event_value')
    #loop over out of frame values
    for i in range(len(beliefSeries_list)-2):
        temp = temp.append(beliefSeries_list[i+2].to_frame(name='event_value'))
    df_1 = temp.append(df_1)
    return df_1


def mean_std_generator(dfnew,df,start,last_start,current_time):
    """
    Creates a list of means and a list of standard deviation based on the differences
    between two belief frames.
    @param dfnew: BeliefsDataFrame
    @param df: BeliefsDataFrame
    @param start: datetime_object
    @param last_start: datetime_object
    @param current_time: datetime_object
    """
    temp_time = start
    last_date = df.iloc[-1].name[0]
    diff_list = []
    mean_list = []
    #loop through range given and calc difference and means
    while temp_time <= last_start:
        if temp_time < last_date:
            diff_list += [abs(get_beliefsSeries_from_event_start(df,temp_time,current_time,'event_value').copy()[0] -
                            get_beliefsSeries_from_event_start(dfnew,temp_time,current_time,'event_value').copy()[0])]
            mean_list += [get_beliefsSeries_from_event_start(dfnew,temp_time,current_time,'event_value').copy()[0]]
        temp_time += df.sensor.event_resolution
    return diff_list,mean_list

#this is from the ridgeline file, so can also be imported etc.
def show_plot(mean, std, start, end, fixedviewpoint=False):
    """
    Creates and shows ridgeline plot

    @param mean: list of mean values
    @param std: list of standard deviation values
    @param start: start hours before event-time
    @param end: end hours before event-time
    @param fixedviewpoint : if true create fixed viewpoint plot
    """
    nr_lines = len(mean)
    x = np.linspace(-10, 30, 500)
    frame = pd.DataFrame()
    for i in range(nr_lines):
        frame["{}".format(i)] = stats.norm.pdf(x, mean[i], std[i])
    pallete = viridis(nr_lines)
    if fixedviewpoint:
        cats = list(frame.keys())
    else:
        cats = list(reversed(frame.keys()))
    source = ColumnDataSource(data=dict(x=x))
    p = figure(y_range=cats, plot_width=900, x_range=(-5, 30), toolbar_location=None)
    for i, cat in enumerate(reversed(cats)):
        y = ridge(cat, frame[cat], 50)
        source.add(y, cat)
        p.patch('x', cat, alpha=0.6, color=pallete[i], line_color="black", source=source)
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
    mapping_dict = {y_ticks[i]: str(y_labels[i]) for i in range(len(y_labels))}
    for i in range(nr_lines):
        if i not in mapping_dict:
            mapping_dict[i]=" "
    mapping_code = "var mapping = {};\n    return mapping[tick];\n    ".format(mapping_dict)
    p.yaxis.formatter = FuncTickFormatter(code=mapping_code)
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
    p.y_range.range_padding = 0.2 / (nr_lines / 168)
    show(p)


# testing
csv_file = 'energy_data.csv'
df = csv_as_belief(csv_file,-9,None)
current_time = datetime.datetime(2015, 1, 1, 16,15, tzinfo=pytz.utc)
start_time = datetime.datetime(2015, 1, 2, 20,45, tzinfo=pytz.utc)
last_start_time = datetime.datetime(2015, 1, 5, 0,0, tzinfo=pytz.utc)
h = main(df,current_time,start_time, last_start_time)
diff_list,mean_list = mean_std_generator(h,df,start_time,last_start_time,current_time)
show_plot(mean_list,diff_list,0,len(mean_list))
