"""Step 4: fair testing + plots.

The golden rule: NEVER let the same patient be in both train and test.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def chance_threshold(n_trials=40, alpha=0.05):
    """Smallest accuracy that beats coin-flipping (one-sided binomial test).
    For 40 trials this is 26/40 = 65%."""
    k = int(binom.isf(alpha, n_trials, 0.5)) + 1
    return k / n_trials


def _run_splits(model, X, y, groups, splitter):
    summary, trials = [], []
    for train_idx, test_idx in splitter.split(X, y, groups):
        m = clone(model)  # brand-new model each round
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        conf = (
            m.predict_proba(X[test_idx]).max(axis=1)
            if hasattr(m, "predict_proba")
            else np.full(len(test_idx), np.nan)
        )
        t = pd.DataFrame(
            {"subject": groups[test_idx], "y_true": y[test_idx],
             "y_pred": pred, "confidence": conf}
        )
        trials.append(t)
        for subj, g in t.groupby("subject"):
            summary.append({
                "subject": subj,
                "n_trials": len(g),
                "balanced_accuracy": balanced_accuracy_score(g.y_true, g.y_pred),
                "mean_confidence": g.confidence.mean(),
            })
    summary = pd.DataFrame(summary)
    trials = pd.concat(trials, ignore_index=True)
    return summary, trials


def leave_one_patient_out(model, X, y, groups):
    """Train on all patients but one, test on the one. Repeat for everyone."""
    return _run_splits(model, X, y, np.asarray(groups), LeaveOneGroupOut())


def grouped_kfold(model, X, y, groups, n_splits=5):
    """Faster debugging version: 5 folds, each patient stays in ONE fold."""
    return _run_splits(model, X, y, np.asarray(groups), GroupKFold(n_splits=n_splits))


def add_chance_flag(summary):
    """Mark which patients are convincingly above chance."""
    out = summary.copy()
    out["above_chance"] = [
        a >= chance_threshold(n) for a, n in zip(out.balanced_accuracy, out.n_trials)
    ]
    return out


# ---------- plots ----------
def plot_patient_scores(summary, title="Balanced accuracy per held-out patient",
                        save_path=None):
    s = summary.sort_values("subject")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(s.subject.astype(str), s.balanced_accuracy, color="#4C78A8")
    ax.axhline(0.5, color="gray", ls="--", label="coin flip (50%)")
    ax.axhline(chance_threshold(int(s.n_trials.median())), color="red", ls=":",
               label="needed to beat chance (p<0.05)")
    ax.axhline(s.balanced_accuracy.mean(), color="green",
               label=f"mean = {s.balanced_accuracy.mean():.3f}")
    ax.set_ylim(0, 1); ax.set_xlabel("Patient"); ax.set_ylabel("Balanced accuracy")
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=90, fontsize=7); plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150)
    return fig


def plot_confusion(trials, labels=("LEFT", "RIGHT"), title="Confusion matrix",
                   save_path=None):
    cm = confusion_matrix(trials.y_true, trials.y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150)
    return fig


def calibration_table(trials, n_bins=5):
    """Does '80% confident' really mean right 80% of the time?"""
    t = trials.dropna(subset=["confidence"]).copy()
    t["correct"] = (t.y_true == t.y_pred).astype(int)
    t["bin"] = pd.cut(t.confidence, np.linspace(0.5, 1.0, n_bins + 1),
                      include_lowest=True)
    return (t.groupby("bin", observed=True)
              .agg(n=("correct", "size"), claimed=("confidence", "mean"),
                   actually_correct=("correct", "mean"))
              .reset_index())


def plot_classifier_comparison(results, save_path=None):
    """results: {model_name: summary_df}"""
    names = list(results)
    means = [results[n].balanced_accuracy.mean() for n in names]
    sds = [results[n].balanced_accuracy.std() for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, yerr=sds, capsize=5, color="#72B7B2")
    ax.axhline(0.5, color="gray", ls="--")
    ax.set_ylim(0, 1); ax.set_ylabel("Mean balanced accuracy (± SD over patients)")
    ax.set_title("Classifier comparison"); plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150)
    return fig
