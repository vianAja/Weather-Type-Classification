from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import xgboost as xgb
import os
os.system('cls')

encoder = LabelEncoder()
full_path = os.getcwd()
path = os.path.join(full_path, 'weather_classification_data.cv')
print(path)

df = pd.read_csv("C:\\Users\LENOVO\Documents\Machine Learning\Weather Type Classification\weather_classification_data.csv")
df['Cloud Cover']   = encoder.fit_transform(df['Cloud Cover'])
df['Location']      = encoder.fit_transform(df['Location'])
df['Season']        = encoder.fit_transform(df['Season'])

y_target = encoder.fit_transform(df["Weather Type"])
dataset = df[[i for i in df.columns[:-1]]]
print(df)

x_train, x_test, y_train, y_test = train_test_split(dataset, 
                                                    y_target,
                                                    test_size=0.2,
                                                    random_state=50)

parameters = {
    'objective':'multi:softmax',
    'num_class': len(df.columns[:-1]),
    'eta': 0.01,
    'max_depth': 6,
}

model = xgb.XGBClassifier(
    parameters
)

model.fit(x_train, y_train)
model.save_model("XGBClassifier.json")

pred = model.predict(x_test)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

pred = encoder.inverse_transform(pred)
y_test = encoder.inverse_transform(y_test)

database_salah = []
database_benar = []
db = []

for prediction, target in zip(pred, y_test):
    data = f'Prediction: {prediction}\tTarget: {target}'
    if prediction != target:
        db.append('Salah')
        database_salah.append(data)
    else:
        db.append('Benar')
        database_benar.append(data)

data = zip(
    np.array(pred),
    np.array(y_test),
    np.array(db)
)

dfPrediction = pd.DataFrame(data, columns=['Prediction', 'Target','Salah/Benar'])
print(f'Jumalah data Precision salah: {len(database_salah)}')
print(f'Jumalah data Precision benar: {len(database_benar)}')