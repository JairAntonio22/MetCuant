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

    scores = [['modelo', 'score']]

    for name, clf in clfs:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append([name, score])
        y_pred = clf.predict(X_test)

    print(tabulate(scores, headers='firstrow', showindex=True))


def main():
    df = pd.read_csv('beer_reviews_updated.csv')
    df = df.head(10000)

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
