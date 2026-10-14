"""Report figures (300 dpi PNG in figures/). Run after experiments.py, models.py, final_model.py."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from features import DEFAULT, FINAL, ROOT, X21, X22, build

FIG, RES = ROOT / "figures", ROOT / "results"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.bbox": "tight", "axes.axisbelow": True})
SCAT = dict(s=14, alpha=.8, edgecolor=SURF, linewidth=.5)
NICE = {"rank_diff": "log FIFA rank diff", "host_diff": "Host diff", "rest_diff": "Rest advantage",
        "prior_goals_scored_diff": "Prior goals scored diff", "prior_goals_conceded_diff": "Prior goals conceded diff",
        "prior_points_per_game_diff": "Prior points/game diff", "prior_clean_sheet_rate_diff": "Prior clean-sheet diff",
        "elite_confederation_diff": "UEFA/CONMEBOL diff", "team_rank": "log team rank", "team_is_host": "Team is host",
        "team_rest": "Team rest days", "team_prior_goals_scored": "Team prior goals scored", "opp_rank": "log opponent rank",
        "opp_goals_conceded_prev_match": "Opp. conceded last match", "is_knockout": "Knockout stage", "matchday": "Matchday"}


def tag(ax, letter):
    t = ax.get_title()
    ax.set_title("")
    ax.set_title(f"({letter}) {t}", loc="left")


def fig_eda():
    raw21, _ = build(DEFAULT)
    d21, d22 = build(FINAL)
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    # (a) goal difference by stage
    ko = d21.stage.str.startswith("Group") == False
    bins = np.arange(-6, 8) - .5
    for i, (mask, name, col) in enumerate([(~ko, "Group stage", BLUE), (ko, "Knockouts", ORANGE)]):
        h, _ = np.histogram(d21.goal_difference[mask], bins)
        ax[0].bar(np.arange(-6, 7) + (i - .5) * .4, h / h.sum(), width=.38, color=col, label=name, edgecolor=SURF, linewidth=.4)
    ax[0].set(title="Goal difference (2.1)", xlabel="Team A − team B goals", ylabel="Share of matches", xlim=(-5.5, 6.5))
    ax[0].set_ylim(0, .42); ax[0].legend(loc="upper left", handlelength=1, ncol=1, fontsize=6.3)
    # (b) goals scored vs Poisson
    g = d22.goals_scored
    k = np.arange(0, 8)
    obs = np.array([(g == v).mean() for v in k])
    ax[1].bar(k, obs, width=.7, color=BLUE, label="Observed", edgecolor=SURF, linewidth=.4)
    ax[1].plot(k, stats.poisson.pmf(k, g.mean()), "o-", color=ORANGE, ms=3.5, lw=1.2, label=f"Poisson(λ={g.mean():.2f})")
    ax[1].set(title="Goals scored (2.2)", xlabel="Goals in match", ylabel="Share of team-matches")
    ax[1].set_ylim(0, .48); ax[1].legend(loc="upper right", handlelength=1.2, fontsize=6.3)
    # (c) ranking vs goal difference, log scale
    r_raw = np.corrcoef(raw21.rank_diff, d21.goal_difference)[0, 1]
    r_log = np.corrcoef(d21.rank_diff, d21.goal_difference)[0, 1]
    jitter = np.random.default_rng(1).normal(0, .08, len(d21))
    ax[2].scatter(d21.rank_diff, d21.goal_difference + jitter, color=BLUE, **SCAT)
    b = np.polyfit(d21.rank_diff, d21.goal_difference, 1)
    xs = np.linspace(d21.rank_diff.min(), d21.rank_diff.max(), 50)
    ax[2].plot(xs, np.polyval(b, xs), color=ORANGE, lw=1.4)
    ax[2].set(title="Rank vs margin", xlabel="log(rank A) − log(rank B)", ylabel="Goal difference")
    ax[2].text(.97, .96, f"r = {r_log:.2f}", transform=ax[2].transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
    # (d) correlation matrix 2.1 predictors + target
    cols = X21 + ["goal_difference"]
    cm = d21[cols].corr().to_numpy()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [BLUE, "#f2f1ee", ORANGE])
    im = ax[3].imshow(cm, cmap=cmap, vmin=-1, vmax=1)
    labels = ["Rank", "Host", "Rest", "Scored", "Conceded", "Pts/game", "Clean sh.", "Conf.", "GD"]
    ax[3].set_xticks(range(len(cols)), labels, rotation=90, fontsize=6)
    ax[3].set_yticks(range(len(cols)), labels, fontsize=6)
    ax[3].grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(cm[i, j]) >= .5:
                ax[3].text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=4.8, color=INK)
    ax[3].set_title("2.1 correlations")
    fig.colorbar(im, ax=ax[3], fraction=.046, pad=.03).ax.tick_params(labelsize=6)
    for a, l in zip(ax, "abcd"):
        tag(a, l)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "fig1_eda.png")


def fig_models():
    fe = pd.read_csv(RES / "feature_engineering_experiments.csv")
    steps = [("Raw features", dict(rank="raw", form="mean_fill", rest="raw")),
             ("+ log rank", dict(rank="log", form="mean_fill", rest="raw")),
             ("+ shrunk form", dict(rank="log", form="shrunk", rest="raw", k=4)),
             ("+ rest sign", dict(rank="log", form="shrunk", rest="sign", k=4))]
    path = []
    for name, c in steps:
        q = (fe.cfg_rank == c["rank"]) & (fe.cfg_form == c["form"]) & (fe.cfg_rest == c["rest"])
        if c["form"] == "shrunk":
            q &= fe.cfg_k == c["k"]
        path.append((name, fe[q].iloc[0]))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.15), gridspec_kw={"width_ratios": [.9, 1.05, 1.05]})
    x = np.arange(len(path))
    for col, task, mk in [(BLUE, "2.1", "o"), (ORANGE, "2.2", "s")]:
        base = path[0][1][f"{task} CV RMSE"]
        v = [(p[f"{task} CV RMSE"] / base - 1) * 100 for _, p in path]
        ax[0].plot(x, v, marker=mk, color=col, lw=1.4, ms=4, label=f"Model {task}")
        ax[0].annotate(f"{v[-1]:.1f}%", (x[-1], v[-1]), xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=6.5, color=INK2)
    ax[0].set_xticks(x, [p[0] for p in path], rotation=25, ha="right")
    ax[0].set(title="Feature engineering (OLS)", ylabel="Change in CV RMSE (%)", xlim=(-.3, 3.7))
    ax[0].axhline(0, color=MUTED, lw=.6)
    ax[0].legend(loc="lower left")
    for a, task in [(ax[1], "21"), (ax[2], "22")]:
        mc = pd.read_csv(RES / f"model_comparison_{task}.csv").sort_values("CV RMSE", ascending=False)
        y = np.arange(len(mc))
        a.errorbar(mc["CV RMSE"], y, xerr=mc["CV RMSE sd"], fmt="o", color=BLUE, ms=4, lw=1, capsize=0,
                   label="Repeated CV (±1 sd)")
        a.scatter(mc["Hold-out RMSE"], y, marker="D", s=16, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=.5,
                  label="Knockout hold-out")
        a.set_yticks(y, mc.Model)
        a.set(title=f"Model {task[0]}.{task[1]} algorithms", xlabel="RMSE (goals, lower is better)")
        best = mc.loc[mc["CV RMSE"].idxmin()]
        a.axvline(best["CV RMSE"], color=MUTED, lw=.6, ls="--")
    h, l_ = ax[1].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=2, bbox_to_anchor=(.68, -.06), fontsize=6.5)
    for a, l in zip(ax, "abc"):
        tag(a, l)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "fig2_models.png")


def fig_final():
    d21, d22 = build(FINAL)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [1.1, 1.1, .85]})
    for a, task, X, col in [(ax[0], "21", X21, BLUE), (ax[1], "22", X22, ORANGE)]:
        c = pd.read_csv(RES / f"final_coefficients_{task}.csv", index_col=0).loc[X].sort_values("std_coef")
        y = np.arange(len(c))
        sig = c.p_value < .05
        a.errorbar(c.std_coef, y, xerr=[c.std_coef - c.std_ci_low, c.std_ci_high - c.std_coef], fmt="none",
                   ecolor=col, lw=1, capsize=0)
        a.scatter(c.std_coef[sig], y[sig.to_numpy()], color=col, s=18, zorder=3, label="p < 0.05")
        a.scatter(c.std_coef[~sig], y[~sig.to_numpy()], facecolor=SURF, edgecolor=col, s=18, zorder=3, label="not significant")
        a.axvline(0, color=MUTED, lw=.6)
        a.set_yticks(y, [NICE[i] for i in c.index])
        a.set(title=f"Model {task[0]}.{task[1]} coefficients", xlabel="Standardised effect, 95% CI")
        a.set_xticks([-0.5, 0], ["−0.5", "0"]); a.set_xlim(-0.95, 0.5)
    h, l_ = ax[0].get_legend_handles_labels()
    import statsmodels.api as sm
    res = sm.OLS(d21.goal_difference, sm.add_constant(d21[X21])).fit()
    ko = (d21.stage.str.startswith("Group") == False).to_numpy()
    ax[2].scatter(res.fittedvalues[~ko], d21.goal_difference[~ko], color=BLUE, label="Group", **SCAT)
    ax[2].scatter(res.fittedvalues[ko], d21.goal_difference[ko], color=ORANGE, marker="D", label="Knockout", **{**SCAT, "s": 12})
    lim = [-5, 6.5]
    ax[2].plot(lim, lim, color=MUTED, lw=.8, ls="--")
    ax[2].set(title="2.1 predicted vs actual", xlabel="Predicted goal difference", ylabel="Actual", xlim=lim, ylim=lim)
    h2, _ = ax[2].get_legend_handles_labels()
    for a, l in zip(ax, "abc"):
        tag(a, l)
    fig.tight_layout(w_pad=1.0)
    fig.legend(h + h2, ["Coefficient p < 0.05", "Not significant", "Group-stage match", "Knockout match"],
               loc="lower center", ncol=4, bbox_to_anchor=(.5, -.07), fontsize=6.5)
    fig.savefig(FIG / "fig3_final.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_eda(); fig_models(); fig_final()
    print("figures written")
