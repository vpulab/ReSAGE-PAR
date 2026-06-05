import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, Optional
import numpy as np

try:
    from sklearn.neighbors import KernelDensity  # type: ignore
    _HAS_SK = True
except Exception:
    KernelDensity = None  # type: ignore
    _HAS_SK = False


def _to_1d(arr: Iterable) -> np.ndarray:
    a = np.asarray(list(arr), dtype=float)
    return a.ravel()


@dataclass
class ThresholdClassifier:
    """Simple 1D threshold classifier trained to separate pos vs neg scores.

    - fit(pos, neg): finds threshold maximizing Youden's J.
    - predict(score): returns 0/1 decision.
    - predict_proba(score): returns pseudo-probability using logistic around threshold.
    """

    threshold: float | None = None

    def fit(self, pos: Iterable, neg: Iterable) -> float:
        pos = _to_1d(pos)
        neg = _to_1d(neg)
        # candidate thresholds between sorted unique scores
        xs = np.unique(np.concatenate([pos, neg]))
        if xs.size == 0:
            self.threshold = 0.0
            return self.threshold
        # evaluate J = TPR - FPR at midpoints
        best_t = xs[0]
        best_j = -1.0
        for t in xs:
            tpr = np.mean(pos >= t) if pos.size else 0.0
            fpr = np.mean(neg >= t) if neg.size else 0.0
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_t = t
        self.threshold = float(best_t)
        return self.threshold

    def predict(self, scores: Iterable) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Classifier not fitted")
        s = _to_1d(scores)
        return (s >= self.threshold).astype(int)

    def predict_proba(self, scores: Iterable) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Classifier not fitted")
        s = _to_1d(scores)
        # simple smooth step using logistic around the threshold
        k = 10.0 / (np.std(s) + 1e-6)
        p = 1.0 / (1.0 + np.exp(-k * (s - self.threshold)))
        return p

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"threshold": self.threshold}, f)

    @classmethod
    def load(cls, path: str) -> "ThresholdClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(threshold=data.get("threshold", 0.0))


def _nanmeanstd(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if arr.size > 1 else 1.0
    return m, (s if s > 0 else 1.0)


def _logpdf_norm(x: Sequence[float], mu: Optional[float], var: Optional[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = 0.0 if mu is None else float(mu)
    v = 1.0 if (var is None or var <= 0) else float(var)
    return -0.5 * (np.log(2.0 * np.pi * v) + ((x - m) ** 2) / v)


@dataclass
class MetricBayesClassifier:
    """Bayesian classifier over 1D scores with Gaussian or KDE models."""

    mode: str = "gauss"  # "gauss" or "kde"
    prior_pos: Optional[float] = None

    mu_pos: Optional[float] = None
    var_pos: Optional[float] = None
    mu_neg: Optional[float] = None
    var_neg: Optional[float] = None
    kde_pos: object = None
    kde_neg: object = None
    llr_threshold: float = 0.0
    _prob_threshold: Optional[float] = None

    def fit(self, pos_train: Sequence[float], neg_train: Sequence[float], bandwidth: str | float = "scott"):
        pos = np.asarray(pos_train, float)
        pos = pos[np.isfinite(pos)]
        neg = np.asarray(neg_train, float)
        neg = neg[np.isfinite(neg)]
        assert pos.size and neg.size, "Need non-empty pos/neg training samples"

        if self.prior_pos is None:
            self.prior_pos = pos.size / float(pos.size + neg.size)

        if self.mode == "gauss":
            self.mu_pos, sd_pos = _nanmeanstd(pos)
            self.var_pos = sd_pos ** 2
            self.mu_neg, sd_neg = _nanmeanstd(neg)
            self.var_neg = sd_neg ** 2
        elif self.mode == "kde":
            if not _HAS_SK:
                raise RuntimeError("KDE requires scikit-learn. Install sklearn or use mode='gauss'.")

            def _scott(x):
                x = np.asarray(x, float)
                x = x[np.isfinite(x)]
                n = max(x.size, 2)
                s = float(np.std(x, ddof=1)) or 1.0
                return max(s * (n ** (-1 / 5)), 1e-6)

            if bandwidth == "scott":
                h_pos = _scott(pos)
                h_neg = _scott(neg)
            else:
                h_pos = float(bandwidth)
                h_neg = float(bandwidth)

            self.kde_pos = KernelDensity(kernel="gaussian", bandwidth=h_pos).fit(pos.reshape(-1, 1))
            self.kde_neg = KernelDensity(kernel="gaussian", bandwidth=h_neg).fit(neg.reshape(-1, 1))
        else:
            raise ValueError("mode must be 'gauss' or 'kde'")

        pi = min(max(float(self.prior_pos), 1e-9), 1 - 1e-9)
        self.llr_threshold = np.log(pi / (1.0 - pi))
        return self

    def _logpdfs(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, float)
        if self.mode == "gauss":
            lp = _logpdf_norm(x, self.mu_pos, self.var_pos)
            ln = _logpdf_norm(x, self.mu_neg, self.var_neg)
        else:
            lp = self.kde_pos.score_samples(x.reshape(-1, 1))
            ln = self.kde_neg.score_samples(x.reshape(-1, 1))
        return lp, ln

    def predict_proba(self, x: Sequence[float]) -> np.ndarray:
        lp, ln = self._logpdfs(x)
        llr = lp - ln + self.llr_threshold
        return 1.0 / (1.0 + np.exp(-llr))

    def predict(self, x: Sequence[float], thresh: Optional[float] = None) -> np.ndarray:
        p = self.predict_proba(x)
        t = self._prob_threshold if (thresh is None and self._prob_threshold is not None) else (0.5 if thresh is None else float(thresh))
        return (p >= t).astype(int)

    def tune_threshold_roc(self, val_scores: Sequence[float], val_labels: Sequence[int]) -> float:
        val_scores = np.asarray(val_scores, float)
        y = np.asarray(val_labels, int)
        mask = np.isfinite(val_scores) & np.isin(y, [0, 1])
        s = val_scores[mask]
        y = y[mask]
        if s.size == 0:
            return self._prob_threshold if self._prob_threshold is not None else 0.5

        p = self.predict_proba(s)
        uniq = np.unique(p)
        bestJ, bestT = -1.0, 0.5
        for t in uniq:
            yhat = (p >= t).astype(int)
            tp = np.sum((y == 1) & (yhat == 1))
            fn = np.sum((y == 1) & (yhat == 0))
            tn = np.sum((y == 0) & (yhat == 0))
            fp = np.sum((y == 0) & (yhat == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            J = tpr + tnr - 1.0
            if J > bestJ:
                bestJ, bestT = float(J), float(t)
        self._prob_threshold = bestT
        return bestT

    def save(self, path: str) -> str:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str) -> "MetricBayesClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected {cls}")
        return obj
