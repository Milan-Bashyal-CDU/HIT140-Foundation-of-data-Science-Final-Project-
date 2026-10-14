"""Compare competing regression algorithms for both tasks on the final feature set.

Every candidate sees the identical outer folds (5-fold x 20 repeats; grouped by match for
2.2). Hyper-parameters are tuned INSIDE each training fold (RidgeCV / LassoCV /
ElasticNetCV / GridSearchCV), i.e. nested cross-validation, so CV scores are not
optimistically biased. A paired corrected resampled t-test (Nadeau & Bengio, 2003)
checks whether each model beats plain OLS.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (ElasticNetCV, HuberRegressor, LassoCV, LinearRegression,
                                  PoissonRegressor, RidgeCV)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import FOLDS, cross_validate, holdout
from features import FINAL, ROOT, X21, X22, build

ALPHAS = np.logspace(-3, 3, 40)


class Subset(LinearRegression):
    """OLS restricted to a subset of columns - used for the ranking-only benchmark."""
    def __init__(self, cols=None):
        super().__init__()
        self.cols = cols

    def fit(self, X, y):
        return super().fit(X[self.cols], y)

    def predict(self, X):
        return super().predict(X[self.cols])


def candidates(task):
    scaled = lambda est: make_pipeline(StandardScaler(), est)
    rank_cols = ["rank_diff"] if task == "2.1" else ["team_rank", "opp_rank"]
    m = {
        "Mean baseline": DummyRegressor(),
        "Rank-only OLS": Subset(rank_cols),
        "OLS": LinearRegression(),
        "Ridge": scaled(RidgeCV(alphas=ALPHAS)),
        "Lasso": scaled(LassoCV(alphas=ALPHAS / 100, cv=5, max_iter=50000)),
        "Elastic Net": scaled(ElasticNetCV(alphas=ALPHAS / 100, l1_ratio=[.2, .5, .8], cv=5, max_iter=50000)),
        "Huber (robust)": scaled(GridSearchCV(HuberRegressor(max_iter=2000),
                                              {"epsilon": [1.35, 1.75, 2.5], "alpha": [1e-3, 1e-1, 1, 10]}, cv=5)),
        "Random forest": GridSearchCV(RandomForestRegressor(n_estimators=100, random_state=0),
                                      {"min_samples_leaf": [5, 15], "max_features": [0.33, 1.0]}, cv=5, n_jobs=-1),
    }
    if task == "2.2":
        m["Poisson GLM"] = scaled(GridSearchCV(PoissonRegressor(max_iter=1000),
                                               {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1]}, cv=5))
    return m


def corrected_ttest(diff, test_frac=1 / FOLDS):
    """Nadeau-Bengio corrected resampled t-test on paired per-fold differences."""
    n = len(diff)
    var = np.var(diff, ddof=1) * (1 / n + test_frac / (1 - test_frac))
    t = np.mean(diff) / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), n - 1)


def compare(task):
    d21, d22 = build(FINAL)
    if task == "2.1":
        df, X, y, groups = d21, X21, "goal_difference", None
        test = d21.stage.str.startswith("Group") == False
    else:
        df, X, y, groups = d22, X22, "goals_scored", "match_id"
        test = d22.is_knockout == 1
    outcome = task == "2.1"
    rows, folds = [], {}
    for name, model in candidates(task).items():
        cv = cross_validate(model, df, X, y, groups=groups, outcome=outcome)
        ho, _, _ = holdout(model, df, X, y, test, outcome=outcome)
        folds[name] = cv
        r = {"Model": name, "CV RMSE": cv.RMSE.mean(), "CV RMSE sd": cv.RMSE.std(),
             "CV MAE": cv.MAE.mean(), "CV R2": cv.R2.mean(),
             "Hold-out RMSE": ho["RMSE"], "Hold-out MAE": ho["MAE"], "Hold-out R2": ho["R2"]}
        if outcome:
            r["CV result acc"], r["Hold-out result acc"] = cv["Result acc"].mean(), ho["Result acc"]
        rows.append(r)
        print(f"  {task} {name:16s} CV RMSE {r['CV RMSE']:.3f}  hold-out RMSE {r['Hold-out RMSE']:.3f}")
    res = pd.DataFrame(rows)
    ols = folds["OLS"].RMSE.to_numpy()
    tests = [corrected_ttest(ols - folds[n].RMSE.to_numpy()) if n != "OLS" else (np.nan, np.nan)
             for n in res.Model]
    res["t vs OLS"], res["p vs OLS"] = zip(*tests)
    res["% folds better than OLS"] = [(folds[n].RMSE.to_numpy() < ols).mean() * 100 for n in res.Model]
    res.round(4).to_csv(ROOT / "results" / f"model_comparison_{task.replace('.', '')}.csv", index=False)
    pd.concat({n: f.RMSE for n, f in folds.items()}, axis=1).to_csv(
        ROOT / "results" / f"cv_fold_rmse_{task.replace('.', '')}.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for task in ["2.1", "2.2"]:
        print(compare(task).round(3).to_string(index=False))
