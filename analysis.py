import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

weather_data_files = {
    "Lahti": "resources/lahti_laune.csv",
    "Oulu": "resources/oulu_pellonpää.csv",
    "Kuopio": "resources/kuopio_savilahti.csv",
    "Jyväskylä": "resources/jyväskylä_lentoasema.csv",
    "Helsinki": "resources/helsinki_kaisaniemi.csv",
    "Joensuu": "resources/joensuu_linnunlahti.csv",
}

# PREPARE SLIP WARNING DATA

slip_warnings = pd.read_json("resources/warnings.json")

slip_warnings = slip_warnings.drop("updated_at", 1) # created_at == updated_at always, so remove duplication
slip_warnings.columns = ["id","city","date"]
# filter out unknown cities:
analyzing_cities = ['Lahti', 'Oulu', 'Kuopio', 'Jyväskylä', 'Helsinki', 'Joensuu']
slip_warnings = slip_warnings[slip_warnings['city'].isin(analyzing_cities)]
# disregard time part in the timestamp => assumption: The warning applies to the day it was given.
slip_warnings['date'] = pd.to_datetime(slip_warnings['date']).dt.date



features = []   # list used for storing features of datapoints
labels = []     # list used for storing labels of datapoints

m = 0    # number of datapoints created so far


for city in analyzing_cities:
    df = pd.read_csv(weather_data_files[city])

    #new_obs = weather_obs[weather_obs["Time"]=="00:00"]
    weather_data = df.assign(date = pd.to_datetime(dict(year=df.Year, month=df.m, day=df.d)))
    # join datasets
    weather_data = weather_data.assign(warning_issued = weather_data.date.isin(slip_warnings[slip_warnings["city"] == city].date))
    # remove columns "year", "month", "day", "time_zone" that are not used 
    data = weather_data.drop(['Year','m','d','Time zone'],axis=1)  #columns are axis 1


    dates = weather_data['date'].unique() 

    # iterate through the list of dates for which we have weather recordings
    for date in dates:
        datapoint = weather_data[(data['date']==date)]  # select weather recordinds corresponding at day "date"

        row_f = datapoint[(datapoint.Time=='00:00')]    # select weather recording at time "01:00"
        row_l = datapoint["warning_issued"]    # select weather recording at time "01:00"
        if len(row_f)==1:
            # TODO: figure out proper features.
            feature = row_f['Air temperature (degC)'].to_numpy()[0]  # extract the temperature recording at "01:00" as feature
            label = row_l    # extract the temperature recording at "12:00" as label
            features.append(feature)                  # add feature to list "features"
            labels.append(label)                      # add label to list "labels"
            m = m+1


X_demo = np.array(features).reshape(m,1)  # convert a list of len=m to a ndarray and reshape it to (m,1)
y_demo = np.array(labels) # convert a list of len=m to a ndarray 

