# Stroke Motor-Imagery Rehabilitation BCI

This project develops an offline EEG decoding pipeline using motor-imagery recordings from 50 acute-stroke patients. EEG recorded during imagined left- and right-hand gripping is filtered to the sensorimotor frequency range (8-30 Hz) and represented using Common Spatial Patterns (CSP). Linear classifiers decode movement intention, with evaluation performed using patient-level splits to test generalization to previously unseen patients. The final model is connected to a simulated rehabilitation interface in which decoded motor intention triggers visual hand-movement feedback.

This is an offline research prototype using previously recorded data and simulated feedback. It is not a clinical or therapeutic device, and it is not a real-time therapy system.

## The project in simple terms

After a stroke, a patient may not be able to move one hand, but the brain can still plan the movement. Patients imagined squeezing their left or right hand while a cap with 29 sensors recorded their brain's electrical activity (EEG). This project:
- cuts the recording into 4-second trials,
- keeps only the brain rhythms linked to movement (8-30 Hz),
- uses CSP to squeeze 29 channels into 6 useful numbers,
- uses a classifier to guess LEFT or RIGHT,
- tests whether this works on a patient the model has never seen,
- shows the guess as an animated virtual hand (simulated feedback).

```
imagined hand movement -> EEG -> 8-30 Hz filter -> CSP -> classifier -> LEFT / RIGHT -> simulated hand feedback
```

## Dataset

- **Liu2024**: EEG motor-imagery recordings from acute-stroke patients.
- **Patients**: 50 acute-stroke patients
- **Electrodes**: 29 EEG channels
- **Trials**: 40 per patient (20 left-hand, 20 right-hand) = 2,000
- **Trial length**: 4 seconds
- **Sampling rate**: 500 Hz originally, resampled to 250 Hz

**Paper**: Liu et al., *An EEG motor imagery dataset for brain computer interface in acute stroke patients*, Scientific Data (2024). https://doi.org/10.1038/s41597-023-02787-8  
Loaded through [MOABB](https://moabb.neurotechx.com/), which downloads the data automatically on first use.  
<!-- Add the exact NEMAR and Figshare links from the dataset pages here. -->
The patient data is not included in this repository. Run the notebook to download it with MOABB.

## Method

| Step | What is done |
|---|---|
| **Epoching** | 4-second windows (0-4 s) per imagery trial |
| **Filtering** | Band-pass 8-30 Hz (mu 8-13 Hz, beta 13-30 Hz) |
| **Resampling** | 500 Hz to 250 Hz |
| **Features** | CSP, 6 components, OAS regularization, log-variance |
| **Classifiers** | LDA, logistic regression, linear SVM (only the classifier changes) |
| **Extension** | Covariance matrices + Riemannian tangent space + logistic regression |

Preprocessing is minimal: filtering, epoching and resampling. There is no artifact rejection, ICA or bad-channel handling, so eye or muscle artifacts remain in the data (see Limitations).

## Evaluation: no data leakage

- **Patient-level splits only.** Leave-one-patient-out: train on 49 patients, test on the 50th, repeat 50 times. No patient appears in both train and test.
- **No leakage.** CSP is supervised, so it sits inside the scikit-learn pipeline and is refit on training patients only in every fold.
- **Metric:** balanced accuracy (average of LEFT recall and RIGHT recall), reported per held-out patient, plus confusion matrices.
- **Chance level.** With 40 trials per patient, about 65% (26 of 40 correct) is needed to beat chance at p < 0.05.
- **Confidence check.** A calibration table compares the model's claimed confidence with how often it is actually right.

## Results

Fill this table from the summary produced by the full 50-patient run. Results from small test runs should not be reported.

| Model | Mean balanced accuracy | SD across patients | Patients above chance |
|---|---|---|---|
| CSP + LDA | - | - | - |
| CSP + Logistic regression | - | - | - |
| CSP + Linear SVM | - | - | - |
| Covariances + Riemannian + LogReg | - | - | - |

| Frequency band (CSP + LDA) | Mean balanced accuracy |
|---|---|
| mu (8-13 Hz) | - |
| beta (13-30 Hz) | - |
| combined (8-30 Hz) | - |

Figures (add from your run): `eeg_example.png`, `filter_effect.png`, `csp_patterns.png`, `patient_scores_all50.png`, `confusion_matrix_all50.png`, `classifier_comparison_all50.png`, `band_comparison_all50.png`, `rehab_demo.png`.

## Simulated rehabilitation demo

A Streamlit app replays recorded trials:
- Choose a held-out patient and a trial. The model is trained on all other patients.
- Press **Decode this trial**.
- The app shows the predicted hand, a confidence score and an animated virtual hand.
- If confidence is below 60%, the app reports "uncertain" and triggers no feedback.
- It also shows how the model performs on all of that patient's trials, so a lucky single trial is not misleading.

Example output:
```
STROKE REHABILITATION BCI
Motor intention detected: RIGHT HAND
Confidence: 87%
Feedback: Virtual right hand closes
```
*(The numbers above are an illustration of the format, not a result.)*

## How to run

### Option A: Google Colab
1. Open a new notebook at colab.research.google.com.
2. Run the notebook cells in order: install, settings, dataset download, epoching, modelling, evaluation.
3. Start with `N_PATIENTS = 3` as a test, then set `N_PATIENTS = None` for all 50 patients.

### Option B: locally
```bash
git clone https://github.com/varun7418/Stroke-Rehabilitation-BCI-From-Real-Patients-ECG.git
cd Stroke-Rehabilitation-BCI-From-Real-Patients-ECG
python -m venv venv
venv\Scripts\activate       # Windows; macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook/stroke_rehab_bci.ipynb
```

Then launch the demo (after the notebook has cached the data):
```bash
# Windows PowerShell:
$env:BCI_CACHE_DIR = "outputs/cache"
# macOS/Linux:
export BCI_CACHE_DIR=outputs/cache

streamlit run app/rehab_demo.py
```

## Repository layout
```
stroke-rehab-bci/
├── README.md
├── requirements.txt
├── notebook/
│   └── stroke_rehab_bci.ipynb
├── app/
│   └── rehab_demo.py
├── figures/
└── results/
```

## Limitations
- Offline replay of recorded EEG with simulated feedback. It does not operate in real time and delivers no therapy.
- Minimal preprocessing: no artifact rejection, ICA or bad-channel handling, so classifiers may partly pick up non-brain artifacts.
- Cross-patient decoding in acute stroke is hard. Expect wide variation between patients, with some near chance.
- Confidence scores can be poorly calibrated and should not be over-trusted.
- The affected-versus-unaffected-hand analysis needs clinical metadata (such as lesion side) that may not be exposed through MOABB. It is not claimed here.

## Possible extensions
- Compare affected vs unaffected hand decoding using the clinical data from the original dataset files.
- Add artifact rejection or ICA and compare results with and without it.
- Try EEGNet or other neural networks after the classical baseline is solid.

## Tech stack
Python, MOABB, MNE-Python, scikit-learn, pyRiemann, NumPy, pandas, SciPy, Matplotlib, Streamlit.

## Citation
If you use the dataset, cite: Liu et al., *An EEG motor imagery dataset for brain computer interface in acute stroke patients*, Scientific Data (2024), https://doi.org/10.1038/s41597-023-02787-8
