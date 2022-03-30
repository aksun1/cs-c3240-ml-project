"""
Machine learning application analysis script.

Predicting slipping weather conditions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 

weather_data_files = {
    "Lahti": "resources/lahti_laune.csv",
    "Oulu": "resources/oulu_pellonpää.csv",
    "Kuopio": "resources/kuopio_savilahti.csv",
    "Jyväskylä": "resources/jyväskylä_lentoasema.csv",
    "Helsinki": "resources/helsinki_kaisaniemi.csv",
    "Joensuu": "resources/joensuu_linnunlahti.csv",
}
analyzing_cities = weather_data_files.keys()

# PREPARE SLIP WARNING DATA

slip_warnings = pd.read_json("resources/warnings.json")

slip_warnings = slip_warnings.drop(columns="updated_at") # created_at == updated_at always, so remove duplication
slip_warnings.columns = ["id","city","date"]
# filter out unknown cities:
slip_warnings = slip_warnings[slip_warnings['city'].isin(analyzing_cities)]
# disregard time part in the timestamp => assumption: The warning applies to the day it was given.
slip_warnings['date'] = pd.to_datetime(slip_warnings['date']).dt.date

features = []   # list used for storing features of datapoints
labels = []     # list used for storing labels of datapoints

frames = []

feature_cols = [
    'Air temperature (degC)', 
    'Maximum temperature (degC)',
    'Snow depth (cm)', 
    'Precipitation amount (mm)'
]

for city in analyzing_cities:
    df = pd.read_csv(weather_data_files[city])

    weather_data = df.assign(date = pd.to_datetime(dict(year=df.Year, month=df.m, day=df.d)))
    # join datasets
    weather_data = weather_data.assign(warning_issued = weather_data.date.isin(slip_warnings[slip_warnings["city"] == city].date))
    # remove columns "year", "month", "day", "time_zone" that are not used 
    weather_data = weather_data.drop(columns=['Year','d','Time zone'])
    # clean wrong values
    weather_data.loc[(weather_data['Snow depth (cm)']==-1), 'Snow depth (cm)'] = 0
    weather_data.loc[(weather_data['Precipitation amount (mm)']==-1), 'Precipitation amount (mm)'] = 0

    weather_data['city'] = city

    frames.append(weather_data)

    dates = weather_data['date'].unique() 
    # iterate through the list of dates for which we have weather recordings
    for date in dates:
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

        if None not in learn_fe_values: # filter out incomplete feature set
            label = day_weather["warning_issued"].to_numpy()[0]    # the warning data is the same for the same date regardless of the index here
            features.append(learn_fe_values)                  # add feature to list "features"
            labels.append(label)                      # add label to list "labels"

X = np.array(features).reshape(len(features),len(features[0]))  # convert a list of len=m to a ndarray and reshape it to (m,*)
y = np.array(labels) # convert a list of len=m to a ndarray 

print(X.shape)
print(feature_cols)

X_model, X_test, y_model, y_test = train_test_split(X, y, test_size=0.1, random_state=41)

# Defining the kfold object we will use for cross validation
kfold = KFold(shuffle=True, random_state=41) # shuffle the ordered data

# study of different ML models:
#classifiers = [DecisionTreeClassifier(max_depth=n, class_weight="balanced") for n in range(1,33)]
#classifiers = [KNeighborsClassifier(n_neighbors=n) for n in range(1,33)]
# final model candidates:
classifiers = [DecisionTreeClassifier(max_depth=9, class_weight="balanced"), KNeighborsClassifier(n_neighbors=5)]

tr_accuracies = {repr(cla): [] for cla in classifiers}
tr_precisions = {repr(cla): [] for cla in classifiers}
tr_recalls = {repr(cla): [] for cla in classifiers}
val_accuracies = {repr(cla): [] for cla in classifiers}
val_precisions = {repr(cla): [] for cla in classifiers}
val_recalls = {repr(cla): [] for cla in classifiers}

# We use the kfold object created earlier, to obtain train and validation sets 
# for k iterations of training and evaluation
for j, (train_indices, val_indices) in enumerate(kfold.split(X_model)): 

    # Define the training and validation data using the indices returned by kfold and numpy indexing 
    X_train, y_train, X_val, y_val = X[train_indices], y[train_indices], X[val_indices], y[val_indices]
    for classifier in classifiers:
        clf_1 = classifier    # initialise a classifier, use default value for all arguments

        clf_1.fit(X_train,y_train)       # fit cfl_1 to data 
        y_pred_train = clf_1.predict(X_train)   # compute predicted labels for training data
        train_accuracy = accuracy_score(y_train, y_pred_train) # compute accuracy on the training set 
        train_precision = precision_score(y_train, y_pred_train) 
        train_recall = recall_score(y_train, y_pred_train) 
        
        y_pred_val = clf_1.predict(X_val)   # compute predicted labels for validation data
        val_accuracy = accuracy_score(y_val, y_pred_val) # compute accuracy on the validation set 
        val_precision = precision_score(y_val, y_pred_val)
        val_recall = recall_score(y_val, y_pred_val)

        tr_accuracies[repr(classifier)].append(train_accuracy) 
        tr_precisions[repr(classifier)].append(train_precision)
        tr_recalls[repr(classifier)].append(train_recall)
        val_accuracies[repr(classifier)].append(val_accuracy)
        val_precisions[repr(classifier)].append(val_precision)
        val_recalls[repr(classifier)].append(val_recall)

print("Train accuracy, Train precision, Train recall, Val accuracy, Val precision, Val recall, Classifier:")
for classifier in tr_accuracies:
    print("{}, {}, {}, {}, {}, {}, {}".format(
        np.mean(tr_accuracies[classifier]),
        np.mean(tr_precisions[classifier]), 
        np.mean(tr_recalls[classifier]), 
        np.mean(val_accuracies[classifier]),
        np.mean(val_precisions[classifier]),
        np.mean(val_recalls[classifier]),
        classifier))

# Compute test error for final selected model
test_classifier = DecisionTreeClassifier(max_depth=9, class_weight="balanced")
test_classifier.fit(X_model,y_model)
test_pred = test_classifier.predict(X_test)
print("Test errors:")
print(accuracy_score(y_test, test_pred), precision_score(y_test, test_pred), recall_score(y_test, test_pred))