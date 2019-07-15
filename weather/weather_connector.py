import requests
import pickle
import pandas as pd
import time
import datetime

API_KEYS_LIMITS = {"82ee012269b6c53f7057fd6d65c99546": 0,
                   "25c04ffd82267cd1d0ecccd8b5113903": 0,
                   "523cb2cab4f753c6885b18b021f1a490": 0,
                   "8254e25ae6fa1aa1e13b10a30e1a44ff": 0,
                   "3ecfcc8a9d368c72f51ac90159e887ab": 0,
                   "3df466b209a233508cae0bbe60c6b3e9": 0}

NEW_YORK_LAT_LNG = "40.730610, -73.935242"


def get_not_full_api():
    for api_key, request_counter in API_KEYS_LIMITS.items():
        if request_counter < 990:
            return api_key


def send_request(timestamp, coordinates=NEW_YORK_LAT_LNG):
    api_key = get_not_full_api()
    if api_key:
        response = requests.get('https://api.darksky.net/forecast/'
                                + api_key + "/"
                                + coordinates + ","
                                # '/40.730610,-73.935242,'
                                + str(timestamp)
                                + "?exclude=currently,flags")
        API_KEYS_LIMITS[api_key] += 1
        return response
    else:
        return None


def load_data():
    with open("failed_and_timestamp3.txt", "rb") as f1:
        failed_and_timestamp = pickle.load(f1)
    last_index = failed_and_timestamp.pop()
    with open("gathered_data.txt", "rb") as f2:
        gathered = pickle.load(f2)
    return last_index, gathered, failed_and_timestamp


def save_data(gathered, last_index, failed):
    failed.append(last_index)
    with open("gathered_data.txt", "wb") as f1:
        pickle.dump(gathered, f1)
    with open("failed_and_timestamp3.txt", "wb") as f2:
        pickle.dump(failed, f2)


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


requests_done = 0
last_index, gathered_data, failed_requests_timestamps = 0, [], []

train_data = load_train("dataset//events_train.csv")


for index, row in train_data.iterrows():
    if index < last_index:
        continue
    response = send_request(row['start_time'], str(row['lat']) + ',' + str(row['lng']))
    if not response:
        print("[INFO] Breaking.")
        break
    elif response.status_code == requests.codes.ok:
        gathered_data.append(response.json())
    else:
        failed_requests_timestamps.append(row['start_time'])
    requests_done += 1
    last_index += 1

    if requests_done % 100 == 0:
        print("[LOG] Finished {rd} requests. {fail} failed. Current index: {ind}"
              .format(rd=requests_done, fail=len(failed_requests_timestamps), ind=last_index))

save_data(gathered_data, last_index, failed_requests_timestamps)
