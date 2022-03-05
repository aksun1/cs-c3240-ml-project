import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # evaluation metrics
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

slip_warnings = slip_warnings.drop(columns="updated_at") # created_at == updated_at always, so remove duplication
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
    weather_data = weather_data.drop(columns=['Year','m','d','Time zone']) 


    dates = weather_data['date'].unique() 
    feature_cols = ['Ground minimum temperature (degC)', 'Air temperature (degC)', 'Precipitation amount (mm)','Snow depth (cm)']
    # iterate through the list of dates for which we have weather recordings
    for date in dates:
        prev_day_weather = weather_data[(weather_data['date']==date - np.timedelta64(1, 'D'))]  # select weather recordinds corresponding at day "date"
        day_weather = weather_data[(weather_data['date']==date)]  # select weather recordinds corresponding at day "date"

        learn_fe_values = []
        for fe in feature_cols:
            feature_column = day_weather[fe].to_numpy()
            feature = None
            # find first non-NaN feature observation from the day's observations
            for feature_candidate in feature_column:
                if not np.isnan(feature_candidate):
                    feature = feature_candidate
                    break
            learn_fe_values.append(feature)
        # for fe in feature_cols:
        #     feature_column = prev_day_weather[fe].to_numpy()
        #     feature = None
        #     # find first non-NaN feature observation from the day's observations
        #     for feature_candidate in feature_column:
        #         if not np.isnan(feature_candidate):
        #             feature = feature_candidate
        #             break
        #     learn_fe_values.append(feature)

        if None not in learn_fe_values:
            label = day_weather["warning_issued"].to_numpy()[0]    # the warning data is the same for the same date regardless of the index here
            features.append(learn_fe_values)                  # add feature to list "features"
            labels.append(label)                      # add label to list "labels"
            m = m+1


X = np.array(features).reshape(m,len(feature_cols))  # convert a list of len=m to a ndarray and reshape it to (m,1)
y = np.array(labels) # convert a list of len=m to a ndarray 

clf_1 = LogisticRegression()    # initialise a LogisticRegression classifier, use default value for all arguments

clf_1.fit(X,y)       # fit cfl_1 to data 
y_pred = clf_1.predict(X)   # compute predicted labels for training data
accuracy = accuracy_score(y, y_pred) # compute accuracy on the training set

print("accuracy of LogReg : ", accuracy)

## sanity check
precision = precision_score(y, y_pred)
print("precision of LogReg : ", precision)
