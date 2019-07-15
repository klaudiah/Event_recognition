import pickle
import pandas as pd
import numpy as np


def column_to_datetime(df, column_name, date_format='%Y-%m-%d %H:%M:%S'):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format=date_format)
    df.dropna(subset=[column_name])
    return df[column_name]


def load_train(path):
    train = pd.read_csv(path).dropna(subset=["lat", "lng"])
    train['start_time'] = column_to_datetime(train, 'start_time')
    train['start_time'] = train['start_time'].map(lambda x: int(x.timestamp()))
    train = train.drop_duplicates(['start_time', 'lat', 'lng'], keep='first')
    return train

with open("gathered_data.txt", "rb") as f2:
    gathered = pickle.load(f2)

train = load_train("'..//dataset//train.csv'")
gat = pd.DataFrame(gathered)
gat['timestamp'] = 0
for gat1, tr1 in zip(gat.iterrows(), train.iterrows()):
    gat.at[gat1[0], 'timestamp'] = tr1[1]['start_time']

gat.to_csv('weather.csv', index=False)
