"""Validation utilities shared by experiments.py and models.py.

Two complementary checks:
  * repeated k-fold cross-validation (5 folds x 20 repeats). For model 2.2 both rows of a
    match always fall in the same fold (grouped CV), so the opponent's row never leaks.
  * a chronological hold-out: train on the 72 group-stage matches, test on the 32
    knockout matches - the closest analogue to genuinely predicting the future.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

REPEATS, FOLDS = 20, 5


def cv_splits(df, groups=None, repeats=REPEATS, k=FOLDS):
    for rep in range(repeats):
        if groups is None:
            yield from KFold(k, shuffle=True, random_state=rep).split(df)
        else:
            g = df[groups].to_numpy()
            ids = np.unique(g)
            perm = dict(zip(ids, np.random.default_rng(rep).permutation(len(ids))))
            yield from GroupKFold(k).split(df, groups=np.vectorize(perm.get)(g))


def metrics(y, p):
    return {"RMSE": mean_squared_error(y, p) ** 0.5, "MAE": mean_absolute_error(y, p),
            "R2": r2_score(y, p)}


def result_accuracy(y, p, band=0.5):
    """Share of matches where predicted outcome (A win / draw / B win) is correct."""
    out = lambda v, b: np.where(v > b, 1, np.where(v < -b, -1, 0))
    return float(np.mean(out(np.asarray(p), band) == out(np.asarray(y), 0)))


def cross_validate(model, df, xcols, y, groups=None, outcome=False):
    rows = []
    for tr, te in cv_splits(df, groups):
        m = clone(model).fit(df.iloc[tr][xcols], df.iloc[tr][y])
        p = m.predict(df.iloc[te][xcols])
        r = metrics(df.iloc[te][y], p)
        if outcome:
            r["Result acc"] = result_accuracy(df.iloc[te][y], p)
        rows.append(r)
    return pd.DataFrame(rows)


def holdout(model, df, xcols, y, test_mask, outcome=False):
    tr, te = df[~test_mask], df[test_mask]
    m = clone(model).fit(tr[xcols], tr[y])
    p = m.predict(te[xcols])
    r = metrics(te[y], p)
    if outcome:
        r["Result acc"] = result_accuracy(te[y], p)
    return r, p, m
