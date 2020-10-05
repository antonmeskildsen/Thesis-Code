import numpy as np


def dominates(y, y_mark):
    y = np.array(y)
    y_mark = np.array(y_mark)
    return np.all(y <= y_mark) and np.any(y < y_mark)


def pareto_frontier(df, columns):
    def each_row(row):
        other_points = df[df['filter'] == row['filter']]
        c = row[columns]
        return not any([dominates(comp[columns], c) for _, comp in other_points.iterrows()])

    df[f'pareto'] = df.apply(each_row, axis=1)
