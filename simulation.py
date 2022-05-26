import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load, hash


def test_models(X):
    clfs = [
        ('mlp reg', load('mlp reg.joblib')),
        ('grad boost', load('gradient boost.joblib')),
    ]

    for name, clf in clfs:
        y_pred = clf.predict(X)

        to_print = X.copy()
        to_print['review overall'] = y_pred

        print(to_print)


def simulate(X):
    test_models(X)

    for column in X.columns:
        low = np.min(X[column])
        high = np.max(X[column])

        simX = X.copy()
        simX[column] = np.random.uniform(low, high, len(X))

        test_models(simX)


def main():
    df = pd.read_csv('beer_reviews_updated.csv')
    df = df.head(10)

    columns_to_drop = [
        'beer_name', 'brewery_id', 'review_time', 'beer_beerid',
        'review_profilename', 'beer_style', 'beer_name',
        'brewery_name'
    ]

    df = df.drop(columns_to_drop, axis=1)
    X = df.drop(['review_overall'], axis=1)

    simulate(X)


if __name__ == '__main__':
    main()
