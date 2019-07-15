import pandas as pd
from sklearn.decomposition import PCA
import json


def column_to_datetime_with_none(df, column_name, date_format='%Y-%m-%d %H:%M:%S'):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format=date_format)
    return df[column_name]


def column_to_datetime(df, column_name, date_format='%Y-%m-%d %H:%M:%S'):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format=date_format)
    df.dropna(subset=[column_name])
    return df[column_name]


def date_to_timestamp(df, column_time):
    df[column_time] = column_to_datetime(df, column_time)
    return df[column_time].map(lambda x: int(x.timestamp()))


def merge_datasets(df1, df2, column_on):
    return df1.merge(df2, on=column_on, how='left')


def city_from_lat_lng(df):
    df = df.sort_values(['lat', 'lng'])
    df = df.reset_index(drop=True)
    city_from_lat_lng_up(df)
    city_from_lat_lng_down(df)


def city_from_lat_lng_up(df):
    df_len = 1806091
    for idx, row in df.iterrows():
        index = df_len-idx
        if pd.isnull(df.iloc[index]['city']) and not pd.isnull(df.iloc[index]['lat']) and not pd.isnull(df.iloc[index]['lng']):
            lat = df.iloc[index]['lat']
            lng = df.iloc[index]['lng']
            if (df.iloc[index-1]['lng'] == lng) & (df.iloc[index-1]['lat'] == lat):
                index_copy = index
                while (df.iloc[index_copy-1]['lng'] == lng) & (df.iloc[index_copy-1]['lat'] == lat):
                    index_copy -= 1
                    if not pd.isnull(df.iloc[index_copy]['city']):
                        for i in range(index_copy, index):
                            df.at[i+1, 'city'] = df.iloc[index_copy]['city']
                        break

                    if not pd.isnull(df.iloc[index_copy]['state']):
                        for i in range(index_copy, index):
                            df.at[i+1, 'state'] = df.iloc[index_copy]['state']
                        break

                    if not pd.isnull(df.iloc[index_copy]['country']):
                        for i in range(index_copy, index):
                            df.at[i+1, 'country'] = df.iloc[index_copy]['country']
                        break


def city_from_lat_lng_down(df):
    for idx, row in df.iterrows():
        if pd.isnull(df.iloc[idx]['city']) and not pd.isnull(df.iloc[idx]['lat']) and not pd.isnull(df.iloc[idx]['lng']):
            lat = df.iloc[idx]['lat']
            lng = df.iloc[idx]['lng']
            if (df.iloc[idx+1]['lng'] == lng) & (df.iloc[idx+1]['lat'] == lat):
                index_copy = idx
                while (df.iloc[index_copy+1]['lng'] == lng) & (df.iloc[index_copy+1]['lat'] == lat):
                    index_copy += 1
                    if not pd.isnull(df.iloc[index_copy]['city']):
                        for i in range(idx, index_copy):
                            df.at[i, 'city'] = df.iloc[index_copy]['city']
                        break

                    if not pd.isnull(df.iloc[index_copy]['state']):
                        for i in range(idx, index_copy):
                            df.at[i, 'state'] = df.iloc[index_copy]['state']
                        break

                    if not pd.isnull(df.iloc[index_copy]['country']):
                        for i in range(idx, index_copy):
                            df.at[i, 'country'] = df.iloc[index_copy]['country']
                        break


def get_notificated_hours_before_event(events_train):
    events_train['notification_time'] = column_to_datetime(events_train, 'notification_time')
    events_train['event_start_time'] = column_to_datetime(events_train, 'event_start_time')
    hours = (events_train['event_start_time'] - events_train['notification_time']).astype('timedelta64[h]')
    return hours


def get_event_start_hour(events):
    events['event_start_time'] = column_to_datetime(events, 'event_start_time')
    start_hours = events['event_start_time'].apply(lambda x: x.hour)
    return start_hours


def get_event_day_of_week(events):
    days = events['event_start_time'].dt.dayofweek
    return days


def get_year(df, column_name):
    df[column_name] = column_to_datetime(df, column_name)
    years = df[column_name].dt.year
    return years


def get_friendship_with_creator(events_users_train, users_friends):
    friendship_list = []
    for idx, row in events_users_train.iterrows():
        user_id = row['user_id']
        creator_id = row['creator_id']
        user_friends = users_friends['friends'][users_friends['user'] == user_id].tolist()[0].split()

        if str(creator_id) in user_friends:
            friendship_list.append(1)
        else:
            friendship_list.append(0)
    return friendship_list


def calculate_distance_between_stems(stem1, stem2):
    return sum((p-q)**2 for p, q in zip(stem1, stem2)) ** .5


def calculate_event_popularity(events_attendees):
    poplarity = events_attendees['yes'].map(lambda x: 0 if pd.isnull(x) else len(x.split()))
    return poplarity


def extract_friends_status(events_train, events_attendees, users_friends):
    friends_attendees_yes = []
    friends_attendees_no = []
    friends_attendees_maybe = []
    friends_attendees_invited = []

    for _, row in events_train.iterrows():
        yes, no, maybe, invited = get_user_friends_status(row, events_attendees, users_friends)
        friends_attendees_yes.append(yes)
        friends_attendees_no.append(no)
        friends_attendees_maybe.append(maybe)
        friends_attendees_invited.append(invited)
    return friends_attendees_yes, friends_attendees_no, friends_attendees_maybe, friends_attendees_invited


def get_user_friends_status(row, events_attendees, users_friends):
    user_id = row['user']
    user_friends = users_friends[users_friends['user'] == user_id]['friends'].map(lambda x: 0 if pd.isnull(x) else len(x.split()))
    event_id = row['event_id']

    users_yes = events_attendees[events_attendees['event'] == event_id]['yes'].map(
        lambda x: 0 if pd.isnull(x) else len(x.split()))
    users_maybe = events_attendees[events_attendees['event'] == event_id]['maybe'].map(
        lambda x: 0 if pd.isnull(x) else len(x.split()))
    users_no = events_attendees[events_attendees['event'] == event_id]['no'].map(
        lambda x: 0 if pd.isnull(x) else len(x.split()))
    users_invited = events_attendees[events_attendees['event'] == event_id]['invited'].map(
        lambda x: 0 if pd.isnull(x) else len(x.split()))

    friends_attendees_yes = list(set(user_friends).intersection(users_yes))
    friends_attendees_no = list(set(user_friends).intersection(users_no))
    friends_attendees_maybe = list(set(user_friends).intersection(users_maybe))
    friends_attendees_invited = list(set(user_friends).intersection(users_invited))

    return len(friends_attendees_yes), len(friends_attendees_no), len(friends_attendees_maybe), len(friends_attendees_invited)


def extract_users_status(event_attendess, train):
    status_list = []
    for idx, row in train.iterrows():
        status = user_status(event_attendess, row['event_id'], row['user_id'])
        status_list.append(status)
    return status_list


def user_status(event_attendess, event_id, user_id):
    event = event_attendess[event_attendess['event'] == event_id]
    if not pd.isnull(event['yes'].tolist()[0]) and user_id in event['yes'].tolist()[0].split():
        return 'yes'
    elif not pd.isnull(event['maybe'].tolist()[0]) and user_id in event['maybe'].tolist()[0].split():
        return 'maybe'
    elif not pd.isnull(event['invited'].tolist()[0]) and user_id in event['invited'].tolist()[0].split():
        return 'invited'
    elif not pd.isnull(event['no'].tolist()[0]) and user_id in event['no'].tolist()[0].split():
        return 'no'
    else:
        return 'none'


def pca(df, num_features):
    pca = PCA(n_components=num_features)
    features = df.iloc[:, 9:110]    # event descriptions features
    return pca.fit_transform(features), pca


def append_wheater(df_train, df_weather, output_path):
    json_daily = json.loads(df_weather['daily'].to_json())
    temperatures = []
    precip = []
    precip_intensity = []

    temperature_len = len("'temperatureHigh\\':")
    precip_len = len("'precipProbability\\':")
    precip_intensity_len = len("'precipIntensity\\':")

    for daily in json_daily:
        text = json_daily[daily]
        if text is not None:
            temperature_start = text.find('temperatureHigh')
            precip_start = text.find('precipProbability')
            precip_intensity_start = text.find('precipIntensity')

            if temperature_start >= 0:
                temperature_end = text.find(",", temperature_start)
                temperature_value = text[temperature_start + temperature_len - 1: temperature_end ]
                temperatures.append(float(temperature_value))
            else:
                temperatures.append(None)

            if precip_start >= 0:
                precip_end = text.find(",", precip_start)
                precip_value = text[precip_start + precip_len - 1: precip_end]
                precip.append(float(precip_value))
            else:
                precip.append(None)

            if precip_intensity_start >= 0:
                precip_intensity_end = text.find(",", precip_intensity_start)
                precip_intensity_value = text[precip_intensity_start + precip_intensity_len - 1: precip_intensity_end]
                precip_intensity.append(float(precip_intensity_value))
            else:
                precip_intensity.append(None)

        else:
            temperatures.append(None)
            precip.append(None)
            precip_intensity.append(None)

    df_weather['temperature'] = temperatures
    df_weather['precip_prob'] = precip
    df_weather['precip_intensity'] = precip_intensity

    df_weather.rename(columns={'latitude': 'lat', 'longitude': 'lng', 'timestamp': 'event_timestamp'}, inplace=True)
    df_merged = merge_datasets(df_train, df_weather[['lat', 'lng', 'event_timestamp', 'temperature', 'precip_prob', 'precip_intensity']], ['lat', 'lng', 'event_timestamp'])

    df_merged.to_csv(output_path)
