"""Reproduce every result and figure in the report, in order.

    python src/run_all.py            # full run (~20-25 min, mostly the random forest)
"""
import time
import warnings

warnings.filterwarnings("ignore")

import experiments, final_model, figures, models
from features import FINAL, build

t0 = time.time()
print("1/5 building datasets ...");            build(FINAL, save=True)
print("2/5 feature-engineering experiments");  experiments.run()
print("3/5 competing algorithms (nested CV)"); [models.compare(t) for t in ["2.1", "2.2"]]
print("4/5 final-model inference");           final_model.run()
print("5/5 figures");                         figures.FIG.mkdir(exist_ok=True); figures.fig_eda(); figures.fig_models(); figures.fig_final()
print(f"done in {(time.time() - t0) / 60:.1f} min - see results/ and figures/")
