import pandas as pd
import numpy as np

desired_width = 520
pd.set_option('display.width', desired_width)

data = pd.read_pickle('data.pkl')
nr_labels = len(data['label'].unique())
nr_samples = data.shape[0]

print(nr_samples)
print(nr_labels)

group = data.groupby(by=['label'])
print(group.agg(['count']))
print(group.agg(['count']).max())
print(group.agg(['count']).min())
print(group.agg(['count']).min())
print(data.groupby(by=['label']).count().mean())


df = pd.read_csv('logs\\test_accuracy.csv')
print(df.groupby(by='model').agg(np.ptp))