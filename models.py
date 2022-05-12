import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)

from sklearn import metrics


def get_error(model, X, y, percent):
    pred = model.predict(X)

    err = np.abs(y - pred) / y
    err = err[err <= percent]

    return len(err) / len(y)


def test_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.80, random_state=42
    )

    clfs = [
        ('lin reg', LinearRegression()),
        ('decision tree', DecisionTreeRegressor()),
        ('svr', SVR()),
        ('mlp reg', MLPRegressor()),
        ('knn reg', KNeighborsRegressor()),
        ('random forest', RandomForestRegressor()),
        ('gradient boost', GradientBoostingRegressor()),
        ('ada boost', AdaBoostRegressor()),
    ]

    scores = [['modelo', 'score', 'err 5%', 'err 10%', 'err 15%']]

    for name, clf in clfs:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        err_5 = get_error(clf, X_test, y_test, 0.05)
        err_10 = get_error(clf, X_test, y_test, 0.10)
        err_15 = get_error(clf, X_test, y_test, 0.15)

        scores.append([name, score, err_5, err_10, err_15])

    print(tabulate(scores, headers='firstrow', showindex=True))


def main():
    df = pd.read_csv('beer_reviews_updated.csv')
    df = df.head(10_000)

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
