"""Step 1 of the pipeline: get the EEG and cut it into trials.

Kid version: EEG is a long wiggly recording of the brain's electricity.
We cut it into 4-second pieces (one piece per "imagine your hand" try)
and keep only the wiggles that are related to movement (8-30 Hz).
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Names used on the rehab screen
DISPLAY_NAMES = {"left_hand": "LEFT HAND", "right_hand": "RIGHT HAND"}


def load_liu2024(subjects=None, fmin=8, fmax=30, tmin=0, tmax=4, resample=250):
    """Download (first time only) and epoch the Liu2024 stroke dataset via MOABB.

    subjects: list of subject numbers, or None for all 50 patients.
              Start small (e.g. [1, 2, 3, 4, 5]) while debugging!
    Returns X (trials x channels x time), y (text labels), metadata (DataFrame
    with a 'subject' column telling which patient each trial came from).
    """
    # imported here so the rest of the project works even without moabb
    from moabb.datasets import Liu2024
    from moabb.paradigms import LeftRightImagery

    dataset = Liu2024()
    paradigm = LeftRightImagery(
        fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=resample
    )
    if subjects is None:
        subjects = dataset.subject_list
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=list(subjects))
    return X, np.asarray(y), metadata.reset_index(drop=True)


def encode_labels(y):
    """Turn 'left_hand'/'right_hand' text into 0/1 numbers.
    Returns the encoded labels and the encoder (so we can go back to text)."""
    enc = LabelEncoder()
    return enc.fit_transform(y), enc


def pretty_label(label):
    """'right_hand' -> 'RIGHT HAND' for the feedback screen."""
    return DISPLAY_NAMES.get(label, str(label).upper().replace("_", " "))


def load_cached(cache_path="cache/liu2024_8-30Hz.npz", **kwargs):
    """Same as load_liu2024 but saves the result so you only download/epoch once."""
    import os
    import pandas as pd
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d["X"], d["y"], pd.DataFrame({"subject": d["subject"]})
    X, y, meta = load_liu2024(**kwargs)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(cache_path, X=X.astype("float32"), y=y,
                        subject=meta["subject"].to_numpy())
    return X, y, meta


def sanity_check(X, y, meta, expected_channels=29):
    """Print the things that must be true BEFORE you trust any result."""
    import pandas as pd
    print("X shape:", X.shape, "(trials, channels, time points)")
    print("Patients:", meta["subject"].nunique(), "| trials per patient:",
          sorted(meta["subject"].value_counts().unique()))
    print("Label counts:", dict(pd.Series(y).value_counts()))
    if X.shape[1] != expected_channels:
        print(f"WARNING: expected {expected_channels} channels but got {X.shape[1]}. "
              "Check for extra non-EEG channels before continuing.")
    if not np.isfinite(X).all():
        print("WARNING: X contains NaN/inf values.")
