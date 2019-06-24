import csv
from os import remove
import datetime
import pytz
from datetime import timedelta
from math import floor
import numpy as np
import timely_beliefs as tb
from timely_beliefs.beliefs.utils import load_time_series
import pandas as pd
from sklearn.linear_model import LinearRegression

### felix code block
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
        #df['2015-05-16 09:14:01+00
    return df


### felix code block

def generator(df, beliefSeries, model):
    """
    Generates a new forecast at a given time using a model

    @param csv_in : csv file containing all forecast data
    @param current_time : datetime string
    @param model : model to use to generate new data
    """

    #check if model is linear_regression
    if isinstance(model, LinearRegression):
        X = np.array([[1], [2], [3], [4]]) # placeholder
        model.fit(X,X)
        return model.predict(np.array([[beliefSeries[0]]]))[0][0]
    #error message
    print("This model functionality has not yet been implemented/ Did you pass the correct model?")
    return observation

def get_beliefsSeries_from_event_start(df,datetime_object,current_time,value):
    return df.loc[datetime_object.strftime("%m/%d/%Y, %H:%M:%S"),value]

def get_beliefsSeries_from_event_start_str(datetime_str,current_time):
    return df.loc[(datetime_str,current_time),'event_value']


def main(df,current_time,start_time,last_start_time = None,model=LinearRegression(), value = 'event_value',addtocsv = False):
    """
    This is the main function of the generator, it opens the data works with the timely_beliefs framework
    and adds results to a timely_beliefs row and/or to the input csvfile

    @param csv_in: csv file containing forecast data
    @param current_time : datetime string
    @param start_time: datetime string
    @param last_start_time: datetime string
    @param model : model to use to generate new data
    @param addtocsv: boolean
    """
    if last_start_time == None:
        last_start_time = start_time

    first_date = df.iloc[0].name[0]
    last_date = df.iloc[-1].name[0]
    #check if current time is in data frame
    if current_time < first_date or current_time > last_date :
        raise SystemExit('Error: your current_time is not in the dataframe')

    #get beliefseries from all the times
    current = get_beliefsSeries_from_event_start(df,current_time,current_time,value)
    start = get_beliefsSeries_from_event_start(df,start_time,current_time,value)
    last_start = get_beliefsSeries_from_event_start(df,last_start_time,current_time,value)

    #create list of beliefSeries
    beliefSeries_list = [start.copy()]
    blfs_list = []
    temp_time = start_time
    i = 0


    #loop over given time slot
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
            beliefSeries_list += [get_beliefsSeries_from_event_start(temp_time,current_time).copy()]
            print(temp_time)
            print(current_time)
            print(get_beliefsSeries_from_event_start(temp_time,current_time).copy())
        temp_time += df.sensor.event_resolution

    print(beliefSeries_list)
    df_1 = tb.BeliefsDataFrame(sensor=df.sensor, beliefs=blfs_list)
    #print(load_time_series(test_list[0],sensor=df.sensor,source='test',belief_horizon=timedelta(hours=0), cumulative_probability=0.5))
    #loops over all time steps
    for beliefSeries in beliefSeries_list:
        if beliefSeries.empty == False:
            beliefSeries[0] = generator(df, current, model)

    if addtocsv == True:
        with open(csv_in, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(datacomp)

    temp = beliefSeries_list[0].to_frame(name=value)
    for i in range(len(beliefSeries_list)-2):
        temp = temp.append(beliefSeries_list[i+2].to_frame(name=value))
    df_1 = temp.append(df_1)

    return df_1

csv_file = 'energy_data.csv'
df = csv_as_belief(csv_file,-9,92)
current_time = datetime.datetime(2015, 1, 1, 16,15, tzinfo=pytz.utc)
start_time = datetime.datetime(2016, 1, 2, 6,15, tzinfo=pytz.utc)
#last_start_time = datetime.datetime(2015, 3, 2, 8,30, tzinfo=pytz.utc)
h = main(df,current_time,start_time,addtocsv=False)
print(h)
