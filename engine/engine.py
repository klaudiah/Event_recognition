from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import data_processing
import numpy as np
import pandas as pd
from surprise import SVD, KNNBasic, CoClustering
from surprise import Dataset
from surprise import Reader


def get_train_and_labels(df):
    x, y = df.drop('interested', 1), df['interested']
    return x, y


def split_dataset(x, y, test_size=0.2, shuffle=True):
    train_x, test_x, train_y, test_y = train_test_split(x,
                                                      y,
                                                      test_size=test_size,
                                                      shuffle=shuffle)
    return train_x, train_y, test_x, test_y


def column_to_numerical_label(df, column_name):
    le = LabelEncoder()
    return le.fit_transform(df[column_name])


def random_forest(train_x, train_y, test_x, test_y, trees_number=10, criterion='gini', max_depth=None, min_sampes_split=2,
                  max_features='auto', bootstrap=True):
    clf = RandomForestClassifier(n_estimators=trees_number,
                                 criterion=criterion,
                                 max_depth=max_depth,
                                 min_samples_split=min_sampes_split,
                                 max_features=max_features,
                                 bootstrap=bootstrap)

    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return accuracy_score(test_y, y_pred), \
           precision_score(test_y, y_pred, average='macro'), \
           recall_score(test_y, y_pred, average='macro'), \
           f1_score(test_y, y_pred, average='macro')


def logist_regrssion(train_x, train_y, test_x, test_y):
    lr = LogisticRegression(solver='liblinear')
    lr.fit(train_x, train_y)
    y_pred = lr.predict(test_x)
    return accuracy_score(test_y, y_pred), \
           precision_score(test_y, y_pred, average='macro'), \
           recall_score(test_y, y_pred, average='macro'), \
           f1_score(test_y, y_pred, average='macro')


def random_forest_crossval(x, y, trees_number=10, criterion='gini', max_depth=None, min_sampes_split=2,
                  max_features='auto', bootstrap=True):
    clf = RandomForestClassifier(n_estimators=trees_number,
                                 criterion=criterion,
                                 max_depth=max_depth,
                                 min_samples_split=min_sampes_split,
                                 max_features=max_features,
                                 bootstrap=bootstrap)

    y_pred = cross_val_predict(clf, x, y)
    return accuracy_score(y, y_pred), \
           precision_score(y, y_pred, average='macro'), \
           recall_score(y, y_pred, average='macro'), \
           f1_score(y, y_pred, average='macro')


def logist_regrssion_crossval(x, y):
    lr = LogisticRegression()
    y_pred = cross_val_predict(lr, x, y)
    return accuracy_score(y, y_pred), \
           precision_score(y, y_pred, average='macro'), \
           recall_score(y, y_pred, average='macro'), \
           f1_score(y, y_pred, average='macro')


def main_crosval():
    df = pd.read_csv('..//dataset//train.csv')
    df['city'] = column_to_numerical_label(df, 'city')
    df['state'] = column_to_numerical_label(df, 'state')
    df['zip'] = column_to_numerical_label(df, 'zip')
    df['country'] = column_to_numerical_label(df, 'country')
    df['locale'] = column_to_numerical_label(df, 'locale')
    df['gender'] = column_to_numerical_label(df, 'gender')
    df['location'] = column_to_numerical_label(df, 'location')

    df = df.drop(
        columns=['event_id', 'user_id', 'event_start_time', 'joinedAt', 'notification_time', 'notificated_before',
                 'friends_yes', 'friends_no', 'friends_maybe', 'friends_invited', 'event_day', 'event_hour',
                 'friendship', 'popularity']
    )

    x, y = get_train_and_labels(df)

    print('### Random Forest ###')
    acc = []
    prec = []
    rec = []
    f1 = []
    for _ in range(0, 100):
        a, p, r, f = random_forest_crossval(x, y)
        acc.append(a)
        prec.append(p)
        rec.append(r)
        f1.append(f)
    print('Acc: {:.1%}'.format(np.mean(acc)), 'Prec: {:.1%}'.format(np.mean(prec)), 'Rec: {:.1%}'.format(np.mean(rec)),
          ('F1: {:.1%}'.format(np.mean(f1))))

    print('### Logistic Regression ###')
    acc = []
    prec = []
    rec = []
    f1 = []
    for _ in range(0, 100):
        a, p, r, f = logist_regrssion_crossval(x, y)
        acc.append(a)
        prec.append(p)
        rec.append(r)
        f1.append(f)
    print('Acc: {:.1%}'.format(np.mean(acc)), 'Prec: {:.1%}'.format(np.mean(prec)), 'Rec: {:.1%}'.format(np.mean(rec)),
          ('F1: {:.1%}'.format(np.mean(f1))))


def collaboravite(data, algo):
    train, test = train_test_split(data, test_size=0.2)

    algo.fit(train)
    res = algo.test(test)
    y_true = []
    y_pred = []
    for u in res:
        y_true.append(u[2])
        y_pred.append(round(u[3]))

    return (accuracy_score(y_true, y_pred), \
           precision_score(y_true, y_pred, average='macro'), \
           recall_score(y_true, y_pred, average='macro'), \
           f1_score(y_true, y_pred, average='macro'))


def main_collaborative():
    df = pd.read_csv('..//dataset//train.csv')
    df = df[['user_id', 'event_id', 'interested']]
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df, reader)
    sim_options = {'name': 'pearson', 'user_based': False}
    algo = KNNBasic(k=10, sim_options=sim_options)

    acc = []
    prec = []
    rec = []
    f1 = []

    for i in range(0, 100):
        a, p, r, f = collaboravite(data, algo)
        acc.append(a)
        prec.append(p)
        rec.append(r)
        f1.append(f)

    print('Acc: {:.1%}'.format(np.mean(acc)), 'Prec: {:.1%}'.format(np.mean(prec)), 'Rec: {:.1%}'.format(np.mean(rec)),
          ('F1: {:.1%}'.format(np.mean(f1))))


if __name__ == "__main__":
    main_collaborative()
