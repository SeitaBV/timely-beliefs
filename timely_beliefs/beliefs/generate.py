import csv
from os import remove
import datetime
from math import floor
import numpy as np
import timely_beliefs as tb
from timely_beliefs.beliefs.utils import load_time_series
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_observation(data,current_time):
    """
    Returns observation at given datetime

    @param data : python list or array containing all forecast data
    @param current_time : datetime string to observe
    """
    #get obeservation from data
    index = get_row(current_time,data)
    return float(data[index][1])

def generator(data, current_time, model):
    """
    Generates a new forecast at a given time using a model

    @param data : python list or array containing all forecast data
    @param current_time : datetime string
    @param model : model to use to generate new data
    """
    #get observation
    observation = get_observation(data, current_time)
    #check if model is linear_regression
    if isinstance(model, LinearRegression):
        X = np.array([[1], [2], [3], [4]]) # placeholder
        model.fit(X,X)
        return model.predict(np.array([[observation]]))[0][0]
    #error message
    print("This model functionality has not yet been implemented/ Did you pass the correct model?")
    return observation


def main(csv_in,current_time,start_time,last_start_time,model=LinearRegression(), addtocsv = False):
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
    #opens data
    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        datacomp = list(csv_reader)
        data = datacomp[1:]
    #gets indexes
    index_current = get_row(current_time,data)
    index_start = get_row(start_time,data)
    index_last_start = get_row(last_start_time,data)
    #creates a temporary csv file
    with open('temp.csv', 'w') as csvfile0:
        writer = csv.writer(csvfile0, delimiter=',')
        index_list = range(index_start,index_last_start+1)
        result = []
        #loops over all time steps
        for index in index_list:
            value = generator(data, current_time, model)
            result += [[data[index][0],data[index_current][0],'Test',0.5,value]]
            #changes values in data list if addtocsv is on
            if addtocsv == True:
                datacomp[index][round((index-index_current)*0.15)] = value
        #builds timely_belief format in csv
        columns = ["event_start","belief_time","source","cumulative_probability","event_value"]
        writer.writerow(columns)
        writer.writerows(result)
    #reads csv as pandas dataframe/timely_belief data frame
    t = pd.read_csv("temp.csv",index_col=0)
    #writes changes to csv file if necessary.
    if addtocsv == True:
        with open(csv_in, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(datacomp)
    remove('temp.csv')
    return t



def get_row(current_time,data):
    """
    Returns index of current_time in data, makes use of binary search.

    @param current_time : datetime string to be searched
    @param data : read csv file
    """
    #convert string to datetime object for comparing
    datetime_object = datetime.datetime.strptime(current_time,'%Y-%m-%d %H:%M:%S')
    print(datetime_object)
    #set left and right halfs
    L = 0
    R = len(data) - 1
    lastindex = 0
    while(L <= R):
        #set middle point/ search index
        index = floor((L+R)/2)
        #round to closest value if exact value not found
        if index == lastindex:
            break
        lastindex = index
        #if time found return
        if datetime.datetime.strptime(data[index][0][:-6],'%Y-%m-%d %H:%M:%S') == datetime_object:
            break
        elif datetime.datetime.strptime(data[index][0][:-6],'%Y-%m-%d %H:%M:%S') > datetime_object:
            if index > 0:
                R = index - 1
        else:
            L = index + 1
    return index

csv_file = 'temperature-linear_regressor-0.5.csv'
h = main(csv_file,"2015-05-16 09:14:01","2015-05-20 09:14:00","2015-05-20 16:30:00",addtocsv=False)
print(h)
