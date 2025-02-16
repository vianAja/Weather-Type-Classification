from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
import pandas as pd

encoder = LabelEncoder()
df = pd.read_csv('weather_classification_data.csv')
df['Cloud Cover']   = encoder.fit_transform(df['Cloud Cover'])
df['Location']      = encoder.fit_transform(df['Location'])
df['Season']        = encoder.fit_transform(df['Season'])

y_target = encoder.fit_transform(df["Weather Type"])
dataset = df[[i for i in df.columns[:-1]]]
print(df)

x_train, x_test, y_train, y_test = train_test_split(dataset, 
                                                    y_target,
                                                    test_size=0.5,
                                                    random_state=50)
model =  xgb.XGBClassifier()
model.load_model('XGBClassifier.json')

pred = model.predict(x_test)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
