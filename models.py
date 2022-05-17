import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
)

from sklearn import metrics

from joblib import dump, load, hash

from tqdm import tqdm
import time


def get_error(model, X, y, percent):
    pred = model.predict(X)
    err = np.abs(y - pred) / y
    err = err[err <= percent]

    return len(err) / len(y)


def test_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=.85, random_state=42
    )

    clfs = [
        ('lin reg', LinearRegression()),
        ('decision tree', DecisionTreeRegressor()),
        ('random forest', RandomForestRegressor(n_jobs=-1)),
        ('knn reg', KNeighborsRegressor(n_jobs=-1)),
        ('mlp reg', MLPRegressor()),
        ('ada boost', AdaBoostRegressor()),
        ('gradient boost', HistGradientBoostingRegressor(max_depth=3, max_leaf_nodes=3)),
    ]

    scores = [['modelo', 'score', 'err 5%', 'err 10%', 'err 15%']]

    for name, clf in tqdm(clfs):
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        
        err_5 = get_error(clf, X_test, y_test, 0.05)
        err_10 = get_error(clf, X_test, y_test, 0.10)
        err_15 = get_error(clf, X_test, y_test, 0.15)
    

        scores.append([name, score, err_5, err_10, err_15])

        dump(clf,"{}.joblib".format(name))
        
        time.sleep(0.5)

    print(tabulate(scores, headers='firstrow', showindex=True))


def n_samples(df,num):
    samples = num
    print("Testing models with {} datapoints which is {:.2f}% of the dataset".format(samples, (samples/len(df)*100)))
    return df.head(samples)

    
def main():
    df = pd.read_csv('beer_reviews_updated.csv')
    df = n_samples(df,800000)

    columns_to_drop = [
        'beer_name', 'brewery_id', 'review_time', 'beer_beerid',
        'review_profilename', 'beer_style', 'beer_name',
        'brewery_name'
    ]

    df = df.drop(columns_to_drop, axis=1)

    X = df.drop(['review_overall'], axis=1)
    y = df['review_overall']

    test_models(X, y)


if __name__ == '__main__':
    main()
