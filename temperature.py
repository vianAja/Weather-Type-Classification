from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor as xgb_reg 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def main(file):
    plt.figure(figsize=(10,10))
    encoder = LabelEncoder()
    df = pd.read_csv(file)
    data_feature = df[[
        'Humidity',
        'Wind Speed',
        'Atmospheric Pressure',
        'UV Index'
    ]]
    print(df)
    print(data_feature)
    
    """
    Initialize the data target
    """
    target = df[[
        'Temperature',
        'Weather Type'
        ]]
    target['Weather Type'] = encoder.fit_transform(target['Weather Type'])

    """
    Initialize the model target
    """
    x_train, x_test, y_train, y_test = train_test_split(
        data_feature,
        target,
        test_size=0.2,
        random_state=25
    )



file = 'weather_classification_data.csv'
main(file)