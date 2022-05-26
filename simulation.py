import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load, hash
from tabulate import tabulate


pdfs = [
    {'distribution': np.random.laplace, 'loc': 4, 'scale': 0.5111},
    {'distribution': np.random.laplace, 'loc': 4, 'scale': 0.4194},
    {'distribution': np.random.laplace, 'loc': 4, 'scale': 0.4893},
    {'distribution': np.random.laplace, 'loc': 4, 'scale': 0.5203},
    {'distribution': np.random.laplace, 'loc': 4, 'scale': 0.5203},
    {'distribution': np.random.normal, 'loc': 7.0424, 'scale': 2.3225},
]


models = [
    ('mlp reg', load('mlp reg.joblib')),
    ('grad boost', load('gradient boost.joblib')),
]


def simulate(X):
    y_mod = [model.predict(X) for _, model in models]
    table = [['variable', 'model', 'change', 'dist from predicted']]

    for i, (name, model) in enumerate(models):
        for sim_mode in ['none', 'complement']:
            simX = X.copy()

            for column, pdf in zip(X.columns, pdfs):
                distribution = pdf['distribution']
                simX[column] = distribution(pdf['loc'], pdf['scale'], len(X))

                if column == 'beer_abv':
                    simX[column] = np.clip(simX[column], 0, 100)

                    if sim_mode == 'complement':
                        simX[column] = 100 - simX[column]
                else:
                    simX[column] = np.clip(simX[column], 1, 5)

                    if sim_mode == 'complement':
                        simX[column] = 4 - (simX[column] - 1) + 1

                y_sim = model.predict(simX)
                dist_from_pred = np.linalg.norm(y_mod[i] - y_sim)
                table.append([column, name, sim_mode, dist_from_pred])

    df = pd.DataFrame(table[1:], columns=table[0])
    max_dist = np.max(np.abs(df['dist from predicted']))
    df['dist from predicted'] = (df['dist from predicted'] / max_dist) * 100
    df.to_csv('sim_results.csv')


def main():
    df = pd.read_csv('beer_reviews_updated.csv')

    columns_to_drop = [
        'beer_name', 'brewery_id', 'review_time', 'beer_beerid',
        'review_profilename', 'beer_style', 'beer_name',
        'brewery_name'
    ]

    df = df.drop(columns_to_drop, axis=1)
    X = df.drop(['review_overall'], axis=1)
    y = df['review_overall']

    simulate(X)


if __name__ == '__main__':
    main()
