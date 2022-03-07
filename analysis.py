import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier  # evaluation metrics
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

frames = []

m = 0    # number of datapoints created so far


for city in analyzing_cities:
    df = pd.read_csv(weather_data_files[city])

    #new_obs = weather_obs[weather_obs["Time"]=="00:00"]
    weather_data = df.assign(date = pd.to_datetime(dict(year=df.Year, month=df.m, day=df.d)))
    # join datasets
    weather_data = weather_data.assign(warning_issued = weather_data.date.isin(slip_warnings[slip_warnings["city"] == city].date))
    # remove columns "year", "month", "day", "time_zone" that are not used 
    weather_data = weather_data.drop(columns=['Year','m','d','Time zone'])
    weather_data['city'] = city

    frames.append(weather_data)


    dates = weather_data['date'].unique() 
    feature_cols = ['Air temperature (degC)', 'Maximum temperature (degC)', 'Snow depth (cm)', 'Precipitation amount (mm)']
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
            # filter outliers:
            if -5 < learn_fe_values[1] < 20:
                label = day_weather["warning_issued"].to_numpy()[0]    # the warning data is the same for the same date regardless of the index here
                features.append(learn_fe_values)                  # add feature to list "features"
                labels.append(label)                      # add label to list "labels"
                m = m+1

#pd.concat(frames).to_excel("labeled_data.xlsx")

X = np.array(features).reshape(m,len(feature_cols))  # convert a list of len=m to a ndarray and reshape it to (m,1)
y = np.array(labels) # convert a list of len=m to a ndarray 

# split the data in to test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifiers = [LogisticRegression(class_weight="balanced"), KNeighborsClassifier(), DecisionTreeClassifier(), SVC(), MLPClassifier()]

for classifier in classifiers:
    clf_1 = classifier    # initialise a LogisticRegression classifier, use default value for all arguments

    clf_1.fit(X_train,y_train)       # fit cfl_1 to data 
    y_pred = clf_1.predict(X_test)   # compute predicted labels for training data
    accuracy = accuracy_score(y_test, y_pred) # compute accuracy on the training set
    precision = precision_score(y_test, y_pred, zero_division=0)

    print("{}   {}  {}".format(accuracy, precision, classifier.__class__.__name__))

