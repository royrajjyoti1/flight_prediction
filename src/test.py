import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import mlflow
from get_data import read_params
import yaml
import sqlite3
import mlflow
import joblib
import argparse


def load_data(path):
    data=pd.read_excel(path)
    return data

def data_cleaning(data):
    print("NA valued present in data: {}".format(data.isnull().sum()))
    data.dropna(inplace=True)
    print("NA dropped")
    print("-"*50)
    print("Duplicate Values in our data: {}".format(data.duplicated().sum()))
    print("Duplicates Dropped")
    data.drop_duplicates(inplace=True)
    return data

def data_preprocessing(data):
    data['Journey_day']=pd.to_datetime(data['Date_of_Journey'],format="%d/%m/%Y").dt.day
    data['Journey_month']=pd.to_datetime(data['Date_of_Journey'],format="%d/%m/%Y").dt.month
    data.drop(["Date_of_Journey"],axis=1,inplace=True)
    data['Dep_hour']=pd.to_datetime(data['Dep_Time']).dt.hour
    data['Dep_min']=pd.to_datetime(data['Dep_Time']).dt.minute

    data.drop(["Dep_Time"],axis=1,inplace=True)
    data['Arrival_hour']=pd.to_datetime(data['Arrival_Time']).dt.hour
    data['Arrival_min']=pd.to_datetime(data['Arrival_Time']).dt.minute

    data.drop(["Arrival_Time"],axis=1,inplace=True)
    duration = list(data["Duration"])

    for i in range(len(duration)):
        if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
            else:
                duration[i] = "0h " + duration[i]           # Adds 0 hour

    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
        duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
    # Adding duration_hours and duration_mins list to train_data dataframe

    data["Duration_hours"] = duration_hours
    data["Duration_mins"] = duration_mins
    data.drop(["Duration"], axis = 1, inplace = True)
    
    Airline = data[["Airline"]]

    Airline = pd.get_dummies(Airline, drop_first= True)
    
    Source = data[["Source"]]

    Source = pd.get_dummies(Source, drop_first= True)
    
    Destination = data[["Destination"]]

    Destination = pd.get_dummies(Destination, drop_first = True)
    data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
    data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
    data_train = pd.concat([data, Airline, Source, Destination], axis = 1)
    data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
    
    return data_train

def traintest_split(data_train):
    X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
    
    y = data_train.iloc[:, 1]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test


def predict(clf,X_test):
    y_pred=clf.predict(X_test)
    return y_pred


def performance_matrix(clf,y_pred,y_test):
    from sklearn import metrics
    acc=clf.score(X_train,y_train)
    print("Accuracy",acc)
    
    mae=metrics.mean_absolute_error(y_test, y_pred)
    print('MAE:', mae)
    
    mse=metrics.mean_squared_error(y_test, y_pred)
    print('MSE:',mse )
    
    rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', rmse)
    
    return {'Accuracy': round(acc,2),'MAE':round(mae,2),'MSE':round(mse,2),'RMSE':round(rmse,2)}


def hyperparameter_tunning(config,X_train,y_train):
    from sklearn.model_selection import RandomizedSearchCV
    n_estimators = config['estimators']['RandomizedSearchCV']['params']['n_estimators']
    max_features = config["estimators"]["RandomizedSearchCV"]["params"]["max_features"]
    max_depth = config["estimators"]["RandomizedSearchCV"]["params"]["max_depth"]
    min_samples_split = config["estimators"]["RandomizedSearchCV"]["params"]["min_samples_split"]
    min_samples_leaf = config["estimators"]["RandomizedSearchCV"]["params"]["min_samples_leaf"]


    
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
    
    
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    
    # Random search of parameters, using 5 fold cross validation, 
    # search across 100 different combinations
    rf_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
    
    rf_random.fit(X_train,y_train)
    best_params=rf_random.best_params_
    
    
    n_estimators = best_params['n_estimators']
    min_samples_split = best_params['min_samples_split']
    min_samples_leaf = best_params['min_samples_leaf']
    max_features = best_params['max_features']
    max_depth = best_params['max_depth']
    
    
    model_tuned = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split,
                                         min_samples_leaf= min_samples_leaf, max_features = max_features,
                                         max_depth= max_depth) 
    model_tuned.fit( X_train, y_train)
    return model_tuned,best_params


def create_experiment(experiment_name,run_name, run_metrics,model, confusion_matrix_path = None, 
                      roc_auc_plot_path = None, run_params=None):
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000 ") #uncomment this line if you want to use any database like sqlite as backend storage for model
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        mlflow.sklearn.log_model(model, "model")
        
        
        
        mlflow.set_tag("tag1", "Random Forest")
        mlflow.set_tags({"tag2":"Randomized Search CV", "tag3":"Production"})
            
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def read_params(config_path):
    with open(config_path,'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


config_path="params.yaml"
config=read_params(config_path)

mlflow_config=config["mlflow_config"]
remote_server_uri=mlflow_config["remote_server_uri"]




train_data=config["data_source"]["s3_source"]
data=load_data(train_data)
cleaned_data=data_cleaning(data)
final_data=data_preprocessing(cleaned_data)
X_train, X_test, y_train, y_test=traintest_split(final_data)


experiment_name = mlflow_config["experiment_name"]
run_name=mlflow_config["run_name"]
model_tuned,best_params = hyperparameter_tunning(config,X_train, y_train)

run_params = best_params

y_pred = predict(model_tuned,X_test) #will return the predicted class
#y_pred_prob = predict_prob_on_test_data(model_tuned,X_test) #model.predict_proba(X_test)
run_metrics = performance_matrix(model_tuned,y_test, y_pred)

create_experiment(experiment_name,run_name,run_metrics,model_tuned,run_params)




