import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("weather_classification_data.csv")
temp = np.array(df['Temperature']).reshape(1, -1).tolist()[0]#[:1000]
humd = np.array(df['Humidity']).reshape(1, -1)   .tolist()[0]#[:1000]
wind = np.array(df['Wind Speed']).reshape(1, -1) .tolist()[0]#[:1000]
hari = [i for i in range(len(wind)+1)][1:]

fig, ax = plt.subplots(3,1, figsize=(20,8))

ax[0].plot(hari, temp, label="Temperature", color='blue')
ax[0].set_title("Temperature")
ax[0].set_xlabel('Hari')
ax[0].set_ylabel("Temperature")
ax[0].legend()

ax[1].plot(hari, humd, label="Humidity", color='green')
ax[1].set_title("Humidity")
ax[1].set_xlabel('Hari')
ax[1].set_ylabel('Humidity')
ax[1].legend()

ax[2].plot(hari, wind, label="Wind Speed", color='yellow')
ax[2].set_title("Wind Speed")
ax[2].set_xlabel('Hari')
ax[2].set_ylabel('Wind Speed')
ax[2].legend()

fig.tight_layout()
plt.show()