"""One command = the whole real experiment.

    python scripts/run_full_evaluation.py --subjects 5     # quick test
    python scripts/run_full_evaluation.py                  # all 50 patients

It downloads Liu2024 through MOABB (first run only, then cached), runs
leave-one-patient-out for 3 models, saves figures + CSVs, and writes the
REAL numbers to results/summary.md. Copy numbers from there, nowhere else.
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT)
os.chdir(ROOT)

from src.evaluation import (add_chance_flag, calibration_table, chance_threshold,  # noqa: E402
                            leave_one_patient_out, plot_classifier_comparison,
                            plot_confusion, plot_patient_scores)
from src.models import get_models  # noqa: E402
from src.preprocessing import encode_labels, load_cached, sanity_check  # noqa: E402


def synthetic(n_sub=6):
    """ONLY for testing that the code runs. Numbers from this are meaningless."""
    import pandas as pd
    rng = np.random.default_rng(0)
    n = n_sub * 40
    y = np.array(["left_hand", "right_hand"] * (n // 2))
    X = rng.standard_normal((n, 29, 1000))
    X[y == "left_hand", 3] *= 1.3
    X[y == "right_hand", 20] *= 1.3
    return X, y, pd.DataFrame({"subject": np.repeat(np.arange(1, n_sub + 1), 40)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=None, help="first N patients (default: all 50)")
    ap.add_argument("--synthetic", action="store_true", help="fake data, just to test the code")
    args = ap.parse_args()

    if args.synthetic:
        X, y_text, meta = synthetic()
        tag = "SYNTHETIC_TEST"
        print("*** SYNTHETIC DATA: results are meaningless ***")
    else:
        subs = None if args.subjects is None else list(range(1, args.subjects + 1))
        tag = "all50" if subs is None else f"first{args.subjects}"
        X, y_text, meta = load_cached(cache_path=f"cache/liu2024_{tag}.npz", subjects=subs)
    sanity_check(X, y_text, meta)

    y, _ = encode_labels(y_text)
    groups = meta["subject"].to_numpy()
    results, trials_by_model = {}, {}
    for name, model in get_models().items():
        t0 = time.time()
        s, t = leave_one_patient_out(model, X, y, groups)
        results[name] = add_chance_flag(s)
        trials_by_model[name] = t
        fn = name.replace(" ", "_").replace("+", "plus")
        s.to_csv(f"results/{fn}_per_patient_{tag}.csv", index=False)
        print(f"{name:18s} mean={s.balanced_accuracy.mean():.3f} "
              f"SD={s.balanced_accuracy.std():.3f}  ({time.time()-t0:.0f}s)")

    best = max(results, key=lambda n: results[n].balanced_accuracy.mean())
    plot_patient_scores(results[best], title=f"{best} ({tag})",
                        save_path=f"figures/patient_scores_{tag}.png")
    plot_confusion(trials_by_model[best], title=best,
                   save_path=f"figures/confusion_matrix_{tag}.png")
    plot_classifier_comparison(results, save_path=f"figures/classifier_comparison_{tag}.png")

    lines = [f"# Results ({tag})", "",
             f"Leave-one-patient-out, {groups.size} trials, {len(np.unique(groups))} patients. "
             f"Chance threshold for 40 trials: {chance_threshold(40):.0%}.", "",
             "| Model | Mean balanced acc. | SD across patients | Patients above chance |",
             "|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r.balanced_accuracy.mean():.3f} | {r.balanced_accuracy.std():.3f} | "
                     f"{int(r.above_chance.sum())}/{len(r)} |")
    lines += ["", f"Best model: **{best}**", "", "Calibration (LogReg confidence vs reality):", "",
              calibration_table(trials_by_model["CSP + LogReg"]).to_markdown(index=False)
              if _has_tabulate() else calibration_table(trials_by_model["CSP + LogReg"]).to_string(index=False)]
    if args.synthetic:
        lines.insert(1, "**SYNTHETIC DATA. DO NOT REPORT THESE NUMBERS.**")
    open(f"results/summary_{tag}.md", "w").write("\n".join(lines))
    print(f"\nSaved results/summary_{tag}.md and figures/*_{tag}.png")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
