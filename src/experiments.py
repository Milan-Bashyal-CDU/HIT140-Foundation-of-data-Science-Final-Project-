"""Feature-engineering experiments (same 8 variables, different transformations).

Each configuration is scored with plain OLS using repeated cross-validation. The winner is
chosen on CV RMSE only; the knockout hold-out is reported but NOT used for selection, so it
stays an honest test.
"""
import itertools
import pandas as pd
from sklearn.linear_model import LinearRegression

from evaluation import cross_validate, holdout
from features import X21, X22, ROOT, build

GRID = [dict(rank=r, form=f, rest=s, k=k)
        for r, s in itertools.product(["raw", "log"], ["raw", "sign"])
        for f, k in [("mean_fill", 0), ("shrunk", 1), ("shrunk", 2), ("shrunk", 4)]]


def label(c):
    form = "mean-fill" if c["form"] == "mean_fill" else f"shrunk k={c['k']:g}"
    return f"rank={c['rank']}, form={form}, rest={c['rest']}"


def run():
    rows = []
    for cfg in GRID:
        d21, d22 = build(cfg)
        cv21 = cross_validate(LinearRegression(), d21, X21, "goal_difference")
        cv22 = cross_validate(LinearRegression(), d22, X22, "goals_scored", groups="match_id")
        h21, _, _ = holdout(LinearRegression(), d21, X21, "goal_difference", d21.stage.str.startswith("Group") == False)
        h22, _, _ = holdout(LinearRegression(), d22, X22, "goals_scored", d22.is_knockout == 1)
        rows.append({"config": label(cfg), **{f"cfg_{k}": v for k, v in cfg.items()},
                     "2.1 CV RMSE": cv21.RMSE.mean(), "2.1 CV R2": cv21.R2.mean(),
                     "2.1 hold-out RMSE": h21["RMSE"],
                     "2.2 CV RMSE": cv22.RMSE.mean(), "2.2 CV R2": cv22.R2.mean(),
                     "2.2 hold-out RMSE": h22["RMSE"]})
    res = pd.DataFrame(rows)
    res["mean CV RMSE rank"] = res[["2.1 CV RMSE", "2.2 CV RMSE"]].rank().mean(axis=1)
    res = res.sort_values("mean CV RMSE rank").reset_index(drop=True)
    (ROOT / "results").mkdir(exist_ok=True)
    res.round(4).to_csv(ROOT / "results" / "feature_engineering_experiments.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    r = run()
    print(r.drop(columns=[c for c in r if c.startswith("cfg_")]).round(3).to_string())
