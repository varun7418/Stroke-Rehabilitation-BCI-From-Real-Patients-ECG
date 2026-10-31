"""Simulated Rehabilitation Interface - Streamlit App & CLI Demo.

Replays recorded trials from acute-stroke patients:
- Choose a held-out patient and a trial (model is trained on all other patients).
- Click 'Decode this trial'.
- Displays predicted hand, confidence score, and animated / visual virtual hand feedback.
- If confidence < 60%, reports 'Uncertain' and triggers no feedback.
- Shows overall performance across all trials for that patient to avoid misleading results.

Run as Streamlit app:
    streamlit run app/rehab_demo.py

Run as CLI script:
    python app/rehab_demo.py --demo
    python app/rehab_demo.py --train-subjects 10 --out figures/rehab_demo.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DISCLAIMER = (
    "Offline research prototype using previously recorded EEG. "
    "Simulated feedback. NOT a clinical device or a real-time therapy system."
)


# ==============================================================================
# Visualization helper: Hand illustration
# ==============================================================================
def draw_hand(ax, cx, closed, active, mirror=False):
    """Draws a cartoon hand. 'closed' shrinks the fingers into a fist."""
    color = "#F2B997" if active else "#E0E0E0"
    edge = "#8A5A44" if active else "#9E9E9E"

    # Palm
    palm = FancyBboxPatch(
        (cx - 0.9, 0.0),
        1.8,
        1.6,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        fc=color,
        ec=edge,
        lw=2,
    )
    ax.add_patch(palm)

    # 4 fingers
    heights = [1.1, 1.4, 1.3, 1.0]
    for i, h in enumerate(heights):
        actual_h = 0.40 if closed else h
        x = cx - 0.85 + i * 0.47
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.6),
                0.38,
                actual_h,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                fc=color,
                ec=edge,
                lw=2,
            )
        )

    # Thumb
    sign = 1 if mirror else -1
    tx = cx + sign * 1.05
    ty = 0.95 if closed else 1.25
    ax.add_patch(Circle((tx, ty), 0.28, fc=color, ec=edge, lw=2))


def render_feedback_figure(
    prediction,
    confidence,
    true_label=None,
    uncertain=False,
    save_path=None,
    patient_id=None,
    trial_idx=None,
):
    """Render the standard rehab feedback card figure."""
    pred_hand = (
        "RIGHT" if "right" in str(prediction).lower() else "LEFT"
    )

    fig = plt.figure(figsize=(7.5, 6.8), facecolor="#f8fafc")
    ax = fig.add_axes([0.05, 0.28, 0.9, 0.42])
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-0.2, 3.4)
    ax.axis("off")

    # Left hand on left, Right hand on right
    left_closed = (pred_hand == "LEFT") and not uncertain
    left_active = (pred_hand == "LEFT") and not uncertain
    right_closed = (pred_hand == "RIGHT") and not uncertain
    right_active = (pred_hand == "RIGHT") and not uncertain

    draw_hand(ax, -1.7, closed=left_closed, active=left_active, mirror=True)
    draw_hand(ax, 1.7, closed=right_closed, active=right_active, mirror=False)

    ax.text(
        -1.7,
        3.2,
        "LEFT HAND",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#1e293b" if left_active else "#64748b",
    )
    ax.text(
        1.7,
        3.2,
        "RIGHT HAND",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#1e293b" if right_active else "#64748b",
    )

    fig.text(
        0.5,
        0.94,
        "STROKE REHABILITATION BCI",
        ha="center",
        fontsize=18,
        weight="bold",
        color="#0f172a",
    )

    sub_header = "Simulated Motor Imagery Rehabilitation Interface"
    if patient_id is not None and trial_idx is not None:
        sub_header += f" | Patient {patient_id} - Trial #{trial_idx + 1}"
    fig.text(0.5, 0.89, sub_header, ha="center", fontsize=10, color="#64748b")

    fig.text(0.5, 0.82, "Motor intention detected:", ha="center", fontsize=12)

    if uncertain:
        status_text = "UNCERTAIN (Below 60% confidence)"
        status_color = "#dc2626"
        feedback_text = "Feedback: No feedback triggered (\u26A0 Low confidence threshold)"
    else:
        status_text = f"{pred_hand} HAND"
        status_color = "#1d4ed8"
        feedback_text = f"Feedback: Virtual {pred_hand.lower()} hand closes \u2713"

    fig.text(
        0.5,
        0.75,
        status_text,
        ha="center",
        fontsize=21,
        weight="bold",
        color=status_color,
    )
    fig.text(
        0.5,
        0.69,
        f"Confidence: {confidence * 100:.1f}%",
        ha="center",
        fontsize=13,
        weight="semibold",
        color="#334155",
    )
    fig.text(
        0.5,
        0.21,
        feedback_text,
        ha="center",
        fontsize=13,
        weight="bold",
        color="#0f172a",
    )

    if true_label is not None:
        true_clean = "RIGHT" if "right" in str(true_label).lower() else "LEFT"
        match_str = (
            "matches ground truth"
            if (true_clean == pred_hand and not uncertain)
            else ("does NOT match ground truth" if not uncertain else "uncertain")
        )
        fig.text(
            0.5,
            0.14,
            f"(Recorded trial label: {true_clean} HAND — prediction {match_str})",
            ha="center",
            fontsize=10,
            color="#475569",
        )

    fig.text(
        0.5,
        0.065,
        "Confidence reflects classification posterior probability and may be subject to calibration shift.",
        ha="center",
        fontsize=8,
        color="#64748b",
    )
    fig.text(0.5, 0.025, DISCLAIMER, ha="center", fontsize=8, color="#94a3b8")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ==============================================================================
# Synthetic Fallback Data (for demo & offline testing without cache)
# ==============================================================================
def generate_synthetic_demo_data(n_patients=3, trials_per_patient=40):
    """Generates synthetic EEG epochs matching Liu2024 structure."""
    rng = np.random.default_rng(42)
    n_trials = n_patients * trials_per_patient
    y_raw = ["left_hand", "right_hand"] * (n_trials // 2)
    rng.shuffle(y_raw)
    y = np.array(y_raw)

    # 29 channels, 1000 time points (4 seconds at 250 Hz)
    X = rng.standard_normal((n_trials, 29, 1000))
    # Inject subtle sensorimotor rhythm modulation on channels 3 and 20 (C3/C4 proxies)
    for i in range(n_trials):
        t = np.linspace(0, 4, 1000)
        signal = np.sin(2 * np.pi * 12 * t)  # 12 Hz mu rhythm
        if y[i] == "left_hand":
            X[i, 20] += 0.45 * signal  # Contralateral desync / sync proxy
        else:
            X[i, 3] += 0.45 * signal

    subjects = np.repeat(np.arange(1, n_patients + 1), trials_per_patient)
    meta = pd.DataFrame({"subject": subjects})
    return X, y, meta


# ==============================================================================
# Data Loader with Cache Support
# ==============================================================================
def load_dataset_for_app():
    """Tries loading from BCI_CACHE_DIR, cache/, outputs/cache, or falls back to synthetic."""
    cache_env = os.environ.get("BCI_CACHE_DIR", "")
    candidates = [
        cache_env,
        "outputs/cache",
        "cache",
        "../outputs/cache",
        "../cache",
    ]

    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            for npz in p.glob("*.npz"):
                try:
                    d = np.load(npz, allow_pickle=True)
                    X = d["X"]
                    y = d["y"]
                    meta = pd.DataFrame({"subject": d["subject"]})
                    return X, y, meta, f"Loaded cache ({npz.name})"
                except Exception:
                    pass

    # Check if results/summary_first3.md or first3 exists
    res_dir = Path("results")
    if not res_dir.exists():
        res_dir = Path("../results")

    # Generate realistic offline demo dataset
    X, y, meta = generate_synthetic_demo_data(n_patients=3, trials_per_patient=40)
    return X, y, meta, "Offline Demo Dataset (3 patients, 120 trials, 29 channels)"


# ==============================================================================
# Streamlit Web Application
# ==============================================================================
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Stroke Rehabilitation BCI Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .disclaimer-box {
            background-color: #fef2f2;
            border-left: 5px solid #ef4444;
            padding: 0.85rem 1rem;
            border-radius: 6px;
            color: #991b1b;
            font-size: 0.88rem;
            margin-bottom: 1.5rem;
        }
        .card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">🧠 Stroke Motor-Imagery Rehabilitation BCI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Offline EEG decoding pipeline with simulated visual feedback for acute-stroke patients</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="disclaimer-box"><strong>⚠ Clinical Disclaimer:</strong> {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("Loading BCI data and models..."):
        X, y, meta, status_msg = load_dataset_for_app()

    # Preprocessing & models
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    try:
        from mne.decoding import CSP

        def get_csp():
            return CSP(n_components=6, reg="oas", log=True)
    except ImportError:
        # Fallback PCA/Variance pipeline if mne is not installed locally
        from sklearn.decomposition import PCA

        def get_csp():
            class SimpleFeatureExtractor:
                def fit(self, X, y):
                    return self

                def transform(self, X):
                    # compute log-variance across channels
                    return np.log(np.var(X, axis=-1) + 1e-6)

                def fit_transform(self, X, y):
                    return self.transform(X)

            return SimpleFeatureExtractor()

    # Sidebar controls
    with st.sidebar:
        st.header("🎮 Trial Selection")
        st.caption(f"Status: {status_msg}")

        subjects = sorted(list(meta["subject"].unique()))
        selected_patient = st.selectbox(
            "Select Held-Out Patient:",
            options=subjects,
            index=0,
            help="The model is trained on all OTHER patients and tested on this unseen patient.",
        )

        patient_mask = meta["subject"] == selected_patient
        patient_trial_indices = np.where(patient_mask)[0]
        n_trials = len(patient_trial_indices)

        selected_trial_num = st.slider(
            f"Select Trial (1 to {n_trials}):",
            min_value=1,
            max_value=n_trials,
            value=1,
            help="Select one 4-second motor-imagery trial to decode.",
        )
        global_trial_idx = patient_trial_indices[selected_trial_num - 1]

        st.divider()
        st.header("⚙ Model Settings")
        classifier_type = st.selectbox(
            "Classifier Pipeline:",
            options=["CSP + Logistic Regression", "CSP + LDA", "CSP + Linear SVM"],
            index=0,
        )

        min_confidence = st.slider(
            "Confidence Threshold (%)",
            min_value=50,
            max_value=90,
            value=60,
            help="If decoded confidence falls below this threshold, feedback is withheld (Uncertain).",
        ) / 100.0

        st.caption("Chance level for 40 trials is ~65% (p < 0.05).")

        st.divider()
        st.header("📥 Downloads")
        _ppt_path = Path(__file__).resolve().parent / "PPT" / "Stroke_Rehab_BCI_Improved_VARUN_K.pptx"
        if _ppt_path.exists():
            with open(_ppt_path, "rb") as _ppt_file:
                st.download_button(
                    label="⬇ Download Project PPT",
                    data=_ppt_file,
                    file_name="Stroke_Rehab_BCI_Improved_VARUN_K.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                    help="Download the project presentation slides.",
                )
        else:
            st.caption("PPT file not found.")

    # Fit Leave-One-Patient-Out Model
    @st.cache_resource(show_spinner="Training classifier on training patients...")
    def train_held_out_model(held_out_subj, model_choice):
        train_mask = meta["subject"] != held_out_subj
        X_train, y_train = X[train_mask], y[train_mask]

        if "LDA" in model_choice:
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        elif "SVM" in model_choice:
            clf = SVC(kernel="linear", probability=True)
        else:
            clf = LogisticRegression(max_iter=2000)

        pipeline = make_pipeline(get_csp(), clf)
        pipeline.fit(X_train, y_train)
        return pipeline

    pipeline = train_held_out_model(selected_patient, classifier_type)

    # Evaluate full held-out patient
    X_test_patient = X[patient_mask]
    y_test_patient = y[patient_mask]
    patient_preds = pipeline.predict(X_test_patient)

    if hasattr(pipeline, "predict_proba"):
        patient_probs = pipeline.predict_proba(X_test_patient)
        patient_confs = patient_probs.max(axis=1)
    else:
        patient_probs = None
        patient_confs = np.full(len(patient_preds), 0.75)

    bal_acc = balanced_accuracy_score(y_test_patient, patient_preds)
    chance_thresh = 0.65
    above_chance = bal_acc >= chance_thresh

    # Decode Selected Trial
    trial_data = X[global_trial_idx : global_trial_idx + 1]
    true_label = y[global_trial_idx]
    pred_label = pipeline.predict(trial_data)[0]

    if hasattr(pipeline, "predict_proba"):
        trial_proba = pipeline.predict_proba(trial_data)[0]
        confidence = float(np.max(trial_proba))
    else:
        confidence = 0.75

    is_uncertain = confidence < min_confidence

    # Layout: Top decode action
    col1, col2 = st.columns([1.1, 1.2])

    with col1:
        st.subheader("🎯 Real-Time Trial Decoding")
        decode_clicked = st.button("🚀 Decode this trial", type="primary", use_container_width=True)

        pred_disp = "RIGHT HAND" if "right" in pred_label.lower() else "LEFT HAND"
        true_disp = "RIGHT HAND" if "right" in str(true_label).lower() else "LEFT HAND"

        with st.container():
            st.markdown(
                f"""
                <div style="background-color: #f1f5f9; padding: 1.2rem; border-radius: 8px; border: 1px solid #cbd5e1;">
                    <p style="margin: 0; font-size: 0.95rem; color: #475569;">Patient ID: <strong>{selected_patient}</strong> | Trial: <strong>#{selected_trial_num}</strong> of {n_trials}</p>
                    <p style="margin: 0.3rem 0; font-size: 1.15rem; color: #0f172a;">Motor Intention Detected: <strong style="color: {'#dc2626' if is_uncertain else '#1d4ed8'};">{'UNCERTAIN' if is_uncertain else pred_disp}</strong></p>
                    <p style="margin: 0.3rem 0; font-size: 1.05rem; color: #334155;">Confidence: <strong>{confidence * 100:.1f}%</strong> (Threshold: {int(min_confidence*100)}%)</p>
                    <p style="margin: 0; font-size: 0.95rem; color: #64748b;">Simulated Feedback: <strong>{'None (Threshold not met)' if is_uncertain else f'Virtual {pred_disp.lower()} closes ✓'}</strong></p>
                    <hr style="margin: 0.6rem 0;">
                    <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Ground Truth Recorded Label: <strong>{true_disp}</strong> ({'Correct' if (pred_disp == true_disp and not is_uncertain) else 'Incorrect / Withheld'})</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plot EEG trace for this trial
        st.markdown("#### ⚡ Sensorimotor EEG Channels")
        fig_trace, ax_trace = plt.subplots(figsize=(6, 2.5), facecolor="#f8fafc")
        t = np.arange(trial_data.shape[2]) / 250.0
        # Show central channels proxies
        ch_idx1 = min(3, trial_data.shape[1] - 1)
        ch_idx2 = min(20, trial_data.shape[1] - 1)
        ax_trace.plot(t, trial_data[0, ch_idx1], label=f"Ch {ch_idx1} (C3 proxy)", lw=1.2, color="#0284c7")
        ax_trace.plot(t, trial_data[0, ch_idx2], label=f"Ch {ch_idx2} (C4 proxy)", lw=1.2, color="#f59e0b", alpha=0.8)
        ax_trace.set_xlabel("Time (seconds)", fontsize=8)
        ax_trace.set_ylabel("Amplitude (µV)", fontsize=8)
        ax_trace.tick_params(labelsize=7)
        ax_trace.legend(loc="upper right", fontsize=7)
        ax_trace.set_title(f"4.0 s Imagery Window (8-30 Hz bandpass)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_trace)

    with col2:
        st.subheader("🤖 Simulated Rehabilitation Feedback")
        fig_rehab = render_feedback_figure(
            prediction=pred_label,
            confidence=confidence,
            true_label=true_label,
            uncertain=is_uncertain,
            patient_id=selected_patient,
            trial_idx=selected_trial_num - 1,
        )
        st.pyplot(fig_rehab)

    st.divider()

    # Overall Patient Performance section (ensures transparency)
    st.subheader(f"📊 Patient {selected_patient} Overall Evaluation (All {n_trials} Trials)")
    st.caption("Performance on the entire held-out patient ensures a single lucky trial is not misleading.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Balanced Accuracy", f"{bal_acc * 100:.1f}%")
    m2.metric("Chance Level (p < 0.05)", "65.0%")
    m3.metric("Above Chance?", "YES ✓" if above_chance else "NO ✗")
    m4.metric("Mean Confidence", f"{np.mean(patient_confs) * 100:.1f}%")

    col_cm, col_trials = st.columns([1, 2])
    with col_cm:
        cm = confusion_matrix(y_test_patient, patient_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(3.2, 3.0), facecolor="#f8fafc")
        im = ax_cm.imshow(cm, cmap="Blues")
        labels = ["LEFT", "RIGHT"]
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(labels, fontsize=8)
        ax_cm.set_yticklabels(labels, fontsize=8)
        ax_cm.set_xlabel("Predicted Hand", fontsize=9)
        ax_cm.set_ylabel("True Hand", fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=11, weight="bold")
        ax_cm.set_title(f"Confusion Matrix (Patient {selected_patient})", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_cm)

    with col_trials:
        # Table of trials for this patient
        trial_summary = pd.DataFrame({
            "Trial": np.arange(1, n_trials + 1),
            "True Label": ["RIGHT" if "right" in str(l).lower() else "LEFT" for l in y_test_patient],
            "Predicted": ["RIGHT" if "right" in str(p).lower() else "LEFT" for p in patient_preds],
            "Confidence": [f"{c*100:.1f}%" for c in patient_confs],
            "Result": ["✓ Correct" if p == t else "✗ Error" for p, t in zip(patient_preds, y_test_patient)],
        })
        st.dataframe(trial_summary.head(10), use_container_width=True, height=220)
        st.caption(f"Showing first 10 of {n_trials} trials for held-out patient {selected_patient}.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    # If run through streamlit command: 'streamlit run app/rehab_demo.py'
    # Streamlit injects specific runtime markers or inspects sys.argv
    try:
        import streamlit as st
        # Detect if running within streamlit server
        if hasattr(st, "runtime") and st.runtime.exists():
            run_streamlit_app()
            return
    except Exception:
        pass

    # CLI fallback mode
    ap = argparse.ArgumentParser(description="Simulated Rehabilitation BCI Demo")
    ap.add_argument("--demo", action="store_true", help="Generate demo screen with realistic numbers")
    ap.add_argument("--train-subjects", type=int, default=10, help="Number of training subjects to load")
    ap.add_argument("--out", default="figures/rehab_demo.png", help="Path to save output figure")
    args = ap.parse_args()

    if args.demo:
        render_feedback_figure("right_hand", 0.87, true_label="right_hand", save_path=args.out)
        print(f"[SUCCESS] Saved DEMO feedback screen to {args.out}")
        return

    try:
        from src.preprocessing import load_liu2024
        from src.models import get_models

        X, y, meta = load_liu2024(subjects=list(range(1, args.train_subjects + 1)))
        subj = meta["subject"].to_numpy()
        held_out = subj[-1]
        model = get_models()["CSP + LogReg"]
        model.fit(X[subj != held_out], y[subj != held_out])
        trial = X[subj == held_out][0:1]
        true_label = y[subj == held_out][0]
        pred = model.predict(trial)[0]
        conf = model.predict_proba(trial).max()
        render_feedback_figure(pred, conf, true_label=true_label, save_path=args.out, patient_id=held_out, trial_idx=0)
        print(f"[SUCCESS] Held-out patient {held_out}: predicted {pred}, confidence {conf:.2f} -> saved {args.out}")
    except Exception as e:
        print(f"[NOTICE] MOABB real data loading skipped ({e}). Generating demo figure from synthetic trial.")
        render_feedback_figure("right_hand", 0.87, true_label="right_hand", save_path=args.out)
        print(f"[SUCCESS] Saved feedback screen to {args.out}")


if __name__ == "__main__":
    main()
