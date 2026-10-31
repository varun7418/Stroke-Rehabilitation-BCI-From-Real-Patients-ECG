"""Step 3: the classifiers (the part that finally says LEFT or RIGHT).

Everything is a scikit-learn Pipeline: CSP -> classifier (or Covariance -> TangentSpace -> classifier).
Because feature extractors live INSIDE the pipeline, they are re-learned using only the training patients each
time. That prevents "data leakage" (peeking at the test answers).
"""
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from .features import make_csp


def get_models(n_components=6, include_riemannian=True):
    """The models we compare.
    - CSP + LDA
    - CSP + Logistic regression
    - CSP + Linear SVM
    - Covariances + Riemannian + LogReg (extension)
    """
    models = {
        "CSP + LDA": make_pipeline(
            make_csp(n_components),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
        "CSP + LogReg": make_pipeline(
            make_csp(n_components), LogisticRegression(max_iter=2000)
        ),
        # probability=True is needed so the rehab screen can show a confidence
        "CSP + Linear SVM": make_pipeline(
            make_csp(n_components), SVC(kernel="linear", probability=True)
        ),
    }

    if include_riemannian:
        try:
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace

            models["Covariances + Riemannian + LogReg"] = make_pipeline(
                Covariances(estimator="lwf"),
                TangentSpace(metric="riemann"),
                LogisticRegression(max_iter=2000),
            )
        except ImportError:
            pass

    return models


def fresh(model):
    """A new, untrained copy of a model (use one per fold!)."""
    return clone(model)
