# FIFA World Cup 2026 — Linear Regression 2.1 & 2.2

Two linear regression models built on the 104 matches of the 2026 FIFA World Cup:

| Model | Rows | Target | Explanatory variables |
|---|---|---|---|
| **LR 2.1** | 104 matches | Goal difference (team A − team B) | 8 pre-match variables |
| **LR 2.2** | 208 team-match rows | Goals scored by a team | 8 pre-match variables (4 shared with LR 2.1) |

Every explanatory variable is known **before kick-off**:
- the FIFA ranking published on 11 June 2026,
- fixed facts (host nation, confederation, date, stage),
- results of a team's *earlier* matches only.

## Repository layout
```
data/raw/        wc2026_matches_104.csv        104 results, each row links its FIFA match-centre page
                 fifa_ranking_2026-06-11.csv   FIFA ranking at kick-off of the tournament
data/processed/  lr21_goal_difference.csv      final LR 2.1 dataset (match_id + 8 variables + target)
                 lr22_goals_scored.csv         final LR 2.2 dataset (row_id + 8 variables + target)
src/             features.py      builds both datasets; feature-engineering options
                 evaluation.py    repeated / grouped CV and knockout hold-out
                 experiments.py   16 feature-engineering configurations
                 models.py        competing algorithms with nested CV + corrected t-tests
                 final_model.py   OLS inference, diagnostics, Poisson GLM, drop-column importance
                 figures.py       report figures
                 run_all.py       reproduces everything in order
notebooks/       analysis.ipynb   executed walkthrough of the whole analysis
results/         CSV outputs of every experiment and model
figures/         report figures (PNG, 300 dpi)
```

## How to run
```bash
pip install -r requirements.txt
python src/features.py   # build the two datasets only (seconds)
python src/run_all.py    # every experiment, model and figure (~20–25 min)
```
Or open `notebooks/analysis.ipynb`. It loads the slow results from `results/` unless you set `RERUN = True`.

## Variables
| # | LR 2.1 (team A − team B) | LR 2.2 (team-level) |
|---|---|---|
| 1 *(shared)* | log FIFA rank difference | log team FIFA rank |
| 2 *(shared)* | host-nation difference | team is host |
| 3 *(shared)* | rest advantage (−1/0/+1) | rest days (capped at 6) |
| 4 *(shared)* | prior goals scored, difference | team prior goals scored |
| 5 | prior goals conceded, difference | log opponent FIFA rank |
| 6 | prior points per game, difference | opponent goals conceded in previous match |
| 7 | prior clean-sheet rate, difference | knockout-stage indicator |
| 8 | UEFA/CONMEBOL difference | matchday |

The processed datasets also hold ID and label columns that are **not** explanatory variables: `match_id`, `stage`, `team_a` and `team_b` in LR 2.1, and `row_id`, `match_id`, `stage`, `team` and `opponent` in LR 2.2. `stage` is used only to define the knockout hold-out.

"Prior" form variables are shrunk toward the tournament average known before the match (empirical Bayes, k = 4). Before the first match, the prior falls back to the Qatar 2022 average of 172 goals in 64 matches.

## Key results
| | LR 2.1 | LR 2.2 |
|---|---|---|
| OLS R² / adjusted R² (full data) | 0.46 / 0.41 | 0.27 / 0.24 |
| OLS CV RMSE (100 folds) | 1.619 | 1.251 |
| Best model (CV RMSE) | Ranking-only OLS, 1.582 | Ridge, 1.242 (8 variables) |
| Knockout hold-out RMSE of recommended model | 1.393 (72% correct results) | 1.109 |

## Data sources
- **Results:** FIFA match centre, https://www.fifa.com/en/match-centre/ (one link per row in `data/raw/wc2026_matches_104.csv`)
- **FIFA ranking, 11 June 2026:** full list from Sporting News via https://sports.yahoo.com/articles/fifa-world-rankings-2026-updated-150253426.html, cross-checked against https://ipt.fifa.com/news/argentina-reclaim-top-spot-mens-world-ranking-fifa-coca-cola
- **Qatar 2022 goal total:** FIFA Annual Report 2022, https://publications.fifa.com/en/annual-report-2022/2022-at-a-glance/fifa-world-cup-qatar-2022-summary/
