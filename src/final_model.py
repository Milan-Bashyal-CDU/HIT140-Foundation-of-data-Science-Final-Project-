"""Inference on the final models (full data) + feature importance.

* OLS fits with coefficient tables, standardised coefficients and 95% CIs
* assumption checks: Shapiro-Wilk, Breusch-Pagan, Durbin-Watson, VIF
* model 2.2: match-clustered standard errors and a Poisson GLM with a dispersion check
* drop-column importance: rise in CV RMSE when each variable is removed
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from evaluation import cross_validate
from features import FINAL, ROOT, X21, X22, build

OUT = ROOT / "results"


def ols(df, X, y, cluster=None):
    Xc = sm.add_constant(df[X])
    if cluster is None:
        return sm.OLS(df[y], Xc).fit()
    return sm.OLS(df[y], Xc).fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})


def coef_table(res, df, X, y):
    ci = res.conf_int()
    sd_x, sd_y = df[X].std(), df[y].std()
    t = pd.DataFrame({"coef": res.params, "std_err": res.bse, "p_value": res.pvalues,
                      "ci_low": ci[0], "ci_high": ci[1]})
    t["std_coef"] = (t.coef * sd_x / sd_y).reindex(t.index)
    t["std_ci_low"] = t.ci_low * sd_x / sd_y
    t["std_ci_high"] = t.ci_high * sd_x / sd_y
    return t


def diagnostics(res, df, X):
    Xc = sm.add_constant(df[X])
    vif = {f"VIF {c}": variance_inflation_factor(Xc.values, i + 1) for i, c in enumerate(X)}
    return {"n": int(res.nobs), "R2": res.rsquared, "Adj R2": res.rsquared_adj,
            "F p-value": res.f_pvalue, "AIC": res.aic,
            "Shapiro-Wilk p": stats.shapiro(res.resid).pvalue,
            "Breusch-Pagan p": het_breuschpagan(res.resid, res.model.exog)[1],
            "Durbin-Watson": durbin_watson(res.resid), "Max VIF": max(vif.values()), **vif}


def drop_column_importance(df, X, y, groups=None):
    base = cross_validate(LinearRegression(), df, X, y, groups=groups).RMSE.mean()
    rows = []
    for c in X:
        cols = [x for x in X if x != c]
        rmse = cross_validate(LinearRegression(), df, cols, y, groups=groups).RMSE.mean()
        rows.append({"variable": c, "CV RMSE without it": rmse, "RMSE increase": rmse - base})
    return pd.DataFrame(rows).sort_values("RMSE increase", ascending=False)


def run():
    d21, d22 = build(FINAL, save=True)
    out = {}
    r21 = ols(d21, X21, "goal_difference")
    r22 = ols(d22, X22, "goals_scored")
    r22c = ols(d22, X22, "goals_scored", cluster="match_id")
    c21, c22 = coef_table(r21, d21, X21, "goal_difference"), coef_table(r22, d22, X22, "goals_scored")
    c22["p_value_clustered"] = r22c.pvalues
    pois = sm.GLM(d22.goals_scored, sm.add_constant(d22[X22]), family=sm.families.Poisson()).fit()
    c22["poisson_coef"], c22["poisson_p"] = pois.params, pois.pvalues
    c21.to_csv(OUT / "final_coefficients_21.csv"); c22.to_csv(OUT / "final_coefficients_22.csv")
    diag = pd.DataFrame({"2.1 OLS": diagnostics(r21, d21, X21), "2.2 OLS": diagnostics(r22, d22, X22)})
    diag.loc["Poisson dispersion (Pearson chi2/df)", "2.2 OLS"] = pois.pearson_chi2 / pois.df_resid
    diag.to_csv(OUT / "final_diagnostics.csv")
    i21 = drop_column_importance(d21, X21, "goal_difference")
    i22 = drop_column_importance(d22, X22, "goals_scored", groups="match_id")
    i21.to_csv(OUT / "importance_21.csv", index=False); i22.to_csv(OUT / "importance_22.csv", index=False)
    return dict(r21=r21, r22=r22, c21=c21, c22=c22, diag=diag, i21=i21, i22=i22, pois=pois)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    o = run()
    print(o["c21"].round(4)); print(o["c22"].round(4)); print(o["diag"].round(4))
    print(o["i21"].round(4)); print(o["i22"].round(4))
