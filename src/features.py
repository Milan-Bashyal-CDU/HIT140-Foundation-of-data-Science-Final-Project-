"""Feature engineering for the FIFA World Cup 2026 regression models.

Builds the two modelling datasets from the raw files in data/raw:
  * Linear Regression 2.1 - 104 matches, target = goal difference (team A - team B)
  * Linear Regression 2.2 - 208 team-match rows, target = goals scored

Every explanatory variable is available BEFORE kick-off:
  - FIFA ranking published 11 June 2026 (the day the tournament started),
  - fixed facts (host nation, confederation, match date, stage),
  - results of a team's EARLIER World Cup 2026 matches only.

`build(config)` exposes the feature-engineering choices tested in experiments.py:
  rank      : "raw"  -> FIFA rank position
              "log"  -> log(rank); a gap between ranks 1 and 10 matters more than 70 vs 79
  form      : "mean_fill" -> running averages, first match filled with the column mean
              "shrunk"    -> running averages shrunk toward the tournament average known
                             before kick-off (empirical-Bayes style, prior weight `k`)
  rest      : "raw"  -> days since the team's previous match
              "sign" -> -1/0/+1 rest advantage (2.1) or capped rest (2.2), less sensitive
                        to the few unusual gaps
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"

HOSTS = {"Mexico", "USA", "Canada"}
UEFA_CONMEBOL = {
    "Spain", "France", "England", "Portugal", "Netherlands", "Belgium", "Germany", "Croatia",
    "Switzerland", "Türkiye", "Austria", "Norway", "Sweden", "Czechia", "Scotland",
    "Bosnia and Herzegovina",                                                     # UEFA
    "Argentina", "Brazil", "Colombia", "Uruguay", "Ecuador", "Paraguay",          # CONMEBOL
}
STAGE_MATCHDAY = {"Round of 32": 4, "Round of 16": 5, "Quarter-final": 6,
                  "Semi-final": 7, "Third place": 8, "Final": 8}
# Fallback prior before any 2026 match is played: Qatar 2022 averaged 172 goals in 64 matches
PRIOR_GOALS_PER_TEAM = 172 / 64 / 2

X21 = ["rank_diff", "host_diff", "rest_diff", "prior_goals_scored_diff",
       "prior_goals_conceded_diff", "prior_points_per_game_diff",
       "prior_clean_sheet_rate_diff", "elite_confederation_diff"]
X22 = ["team_rank", "team_is_host", "team_rest", "team_prior_goals_scored",
       "opp_rank", "opp_goals_conceded_prev_match", "is_knockout", "matchday"]
SHARED = {"FIFA ranking": ("rank_diff", "team_rank"), "Host nation": ("host_diff", "team_is_host"),
          "Rest days": ("rest_diff", "team_rest"),
          "Prior goals scored": ("prior_goals_scored_diff", "team_prior_goals_scored")}

DEFAULT = dict(rank="raw", form="mean_fill", rest="raw", k=2.0)
FINAL = dict(rank="log", form="shrunk", rest="sign", k=4.0)   # chosen in experiments.py


def load_raw():
    m = pd.read_csv(RAW / "wc2026_matches_104.csv", encoding="utf-8-sig", parse_dates=["date"])
    r = pd.read_csv(RAW / "fifa_ranking_2026-06-11.csv", encoding="utf-8-sig")
    return m.sort_values(["date", "match_id"]).reset_index(drop=True), \
        dict(zip(r.team, r.fifa_rank_2026_06_11))


def team_rows(m):
    """One row per team per match (208 rows), chronological."""
    a = m.assign(team=m.team_a, opponent=m.team_b, gf=m.goals_a, ga=m.goals_b)
    b = m.assign(team=m.team_b, opponent=m.team_a, gf=m.goals_b, ga=m.goals_a)
    t = pd.concat([a, b])[["match_id", "date", "stage", "team", "opponent", "gf", "ga"]]
    t = t.sort_values(["date", "match_id", "team"]).reset_index(drop=True)
    t["pts"] = np.select([t.gf > t.ga, t.gf == t.ga], [3, 1], 0)   # shoot-out = draw
    t["clean"] = (t.ga == 0).astype(int)
    return t


def tournament_prior(t, col, fallback):
    """Average of `col` over all team-matches played strictly BEFORE each row's date."""
    daily = t.groupby("date")[col].agg(["sum", "count"]).cumsum().shift(1)
    mean_before = (daily["sum"] / daily["count"]).reindex(t.date).to_numpy()
    return np.where(np.isnan(mean_before), fallback, mean_before)


def add_history(t, cfg):
    g = t.groupby("team", sort=False)
    t["n_prior"] = g.cumcount()
    fallbacks = {"gf": PRIOR_GOALS_PER_TEAM, "ga": PRIOR_GOALS_PER_TEAM,
                 "pts": 4 / 3, "clean": 0.3}          # neutral pre-tournament priors
    for col in ["gf", "ga", "pts", "clean"]:
        prior_sum = g[col].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
        if cfg["form"] == "shrunk":
            mu = tournament_prior(t, col, fallbacks[col])
            t[f"prior_{col}"] = (prior_sum + cfg["k"] * mu) / (t.n_prior + cfg["k"])
        else:
            avg = prior_sum / t.n_prior.replace(0, np.nan)
            t[f"prior_{col}"] = avg.fillna(avg.mean())
    prev_ga = g["ga"].shift(1)
    if cfg["form"] == "shrunk":
        mu = tournament_prior(t, "ga", PRIOR_GOALS_PER_TEAM)
        t["prev_ga"] = np.where(prev_ga.isna(), mu, (prev_ga.fillna(0) + cfg["k"] * mu) / (1 + cfg["k"]))
    else:
        t["prev_ga"] = prev_ga.fillna(prev_ga.mean())
    rest = g["date"].diff().dt.days
    t["rest"] = rest.fillna(rest.mean())
    t["matchday"] = np.where(t.stage.str.startswith("Group"), t.n_prior + 1,
                             t.stage.map(STAGE_MATCHDAY))
    return t


def build(cfg=None, save=False):
    cfg = {**DEFAULT, **(cfg or {})}
    m, rank = load_raw()
    t = add_history(team_rows(m), cfg)
    t["rank"] = t.team.map(rank)
    t["rank_f"] = np.log(t["rank"]) if cfg["rank"] == "log" else t["rank"]
    t["host"] = t.team.isin(HOSTS).astype(int)
    t["elite"] = t.team.isin(UEFA_CONMEBOL).astype(int)
    t["ko"] = (~t.stage.str.startswith("Group")).astype(int)
    key = t.set_index(["match_id", "team"])
    opp = key.loc[list(zip(t.match_id, t.opponent))].reset_index(drop=True)

    team_rest = t.rest.clip(upper=6) if cfg["rest"] == "sign" else t.rest
    d22 = pd.DataFrame({
        "row_id": np.arange(1, len(t) + 1), "match_id": t.match_id, "stage": t.stage,
        "team": t.team, "opponent": t.opponent,
        "team_rank": t.rank_f, "team_is_host": t.host, "team_rest": team_rest,
        "team_prior_goals_scored": t.prior_gf, "opp_rank": opp.rank_f,
        "opp_goals_conceded_prev_match": opp.prev_ga, "is_knockout": t.ko,
        "matchday": t.matchday, "goals_scored": t.gf})

    A = key.loc[list(zip(m.match_id, m.team_a))].reset_index(drop=True)
    B = key.loc[list(zip(m.match_id, m.team_b))].reset_index(drop=True)
    rest_diff = A.rest - B.rest
    d21 = pd.DataFrame({
        "match_id": m.match_id, "stage": m.stage, "team_a": m.team_a, "team_b": m.team_b,
        "rank_diff": A.rank_f - B.rank_f, "host_diff": A.host - B.host,
        "rest_diff": np.sign(rest_diff) if cfg["rest"] == "sign" else rest_diff,
        "prior_goals_scored_diff": A.prior_gf - B.prior_gf,
        "prior_goals_conceded_diff": A.prior_ga - B.prior_ga,
        "prior_points_per_game_diff": A.prior_pts - B.prior_pts,
        "prior_clean_sheet_rate_diff": A.prior_clean - B.prior_clean,
        "elite_confederation_diff": A.elite - B.elite,
        "goal_difference": m.goals_a - m.goals_b})

    for d, x in [(d21, X21), (d22, X22)]:
        d[x] = d[x].astype(float).round(4)
        assert len(x) == 8 and not d[x].isna().any().any()
    assert len(d21) == 104 and len(d22) == 208 and len(SHARED) <= 4
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        d21.to_csv(PROCESSED / "lr21_goal_difference.csv", index=False, encoding="utf-8-sig")
        d22.to_csv(PROCESSED / "lr22_goals_scored.csv", index=False, encoding="utf-8-sig")
    return d21, d22


if __name__ == "__main__":
    d21, d22 = build(FINAL, save=True)
    print(d21.shape, d22.shape)
