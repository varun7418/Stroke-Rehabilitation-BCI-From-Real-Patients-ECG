"""Step 2: squeeze the EEG into a few useful numbers with CSP.

Kid version: we have 29 microphones (electrodes) listening to the brain.
CSP finds the best *mixes* of microphones, the ones that sound LOUDER for
left-hand thinking and QUIETER for right-hand thinking (or the opposite).
Each trial becomes just 6 numbers.
"""
import mne
from mne.decoding import CSP

mne.set_log_level("WARNING")  # keep notebooks quiet


def make_csp(n_components=6):
    """CSP that is safe for small datasets (reg='oas') and gives log-variance."""
    return CSP(n_components=n_components, reg="oas", log=True)
