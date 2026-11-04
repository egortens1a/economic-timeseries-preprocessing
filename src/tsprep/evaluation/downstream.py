"""Downstream probes and validation-only anomaly calibration."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, r2_score, roc_auc_score, silhouette_score, average_precision_score,
    precision_score, recall_score)


def regression_report(y, pred):
    return dict(mae=float(mean_absolute_error(y, pred)), r2=float(r2_score(y, pred)) if len(y)>1 else None)


def forecast_via_linear_probe(z_train, y_train, z_test, y_test, *, z_val=None, y_val=None):
    candidates = [0.1, 1., 10., 100.]
    if z_val is None:
        candidates = [1.]
    best, score, alpha = None, float("inf"), None
    for a in candidates:
        reg = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(z_train, y_train)
        value = mean_absolute_error(y_val, reg.predict(z_val)) if z_val is not None else 0.
        if value < score:
            best, score, alpha = reg, value, a
    return {**regression_report(y_test, best.predict(z_test)), "ridge_alpha": alpha,
            "validation_mae": float(score) if z_val is not None else None}


def binary_report(y, pred, scores=None):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or not np.isin(y, [0, 1]).all() or not len(y):
        raise ValueError("binary labels and predictions must have equal nonempty shape")
    both = len(np.unique(y)) == 2
    n_positive, n_negative = int(y.sum()), int((y==0).sum())
    return dict(accuracy=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)) if n_positive else None,
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)) if n_positive else None,
        balanced_accuracy=float(balanced_accuracy_score(y, pred)) if both else None,
        roc_auc=float(roc_auc_score(y, scores)) if both and scores is not None else None,
        pr_auc=float(average_precision_score(y, scores)) if both and scores is not None else None,
        false_positive_rate=float(pred[y==0].mean()) if n_negative else None,
        positive_rate_test=float(y.mean()), n_positive=n_positive, n_negative=n_negative,
        note=None if both else "single-class evaluation; sensitivity or specificity is not identifiable")


def classify_via_linear_probe(z_train, y_train, z_test, y_test, class_weight="balanced"):
    tr_ok, te_ok = np.isfinite(y_train), np.isfinite(y_test)
    z_train, y_train = z_train[tr_ok], y_train[tr_ok]
    z_test, y_test = z_test[te_ok], y_test[te_ok]
    if not np.isin(y_train, [0, 1]).all() or not np.isin(y_test, [0, 1]).all():
        raise ValueError("classification probe supports binary targets only")
    if len(np.unique(y_train)) < 2 or not len(y_test):
        return dict(accuracy=None, f1=None, roc_auc=None, pr_auc=None, note="insufficient labelled classes")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=class_weight))
    clf.fit(z_train, y_train)
    scores = clf.predict_proba(z_test)[:, 1]
    return {**binary_report(y_test, (scores >= .5).astype(int), scores),
            "positive_rate_train": float(y_train.mean()), "target": "retrospective_label_at_window_end"}


def _label_windows(labels, window_len, stride, n_windows):
    if min(window_len, stride, n_windows) < 1 or (n_windows-1)*stride+window_len > len(labels):
        raise ValueError("label length does not match windows")
    return [np.asarray(labels)[i*stride:i*stride+window_len] for i in range(n_windows)]


def window_class_labels(labels, window_len, stride, n_windows, rule="last"):
    windows = _label_windows(labels, window_len, stride, n_windows)
    if rule == "last":
        return np.asarray([w[-1] for w in windows])
    if rule != "majority":
        raise ValueError("Unknown window label rule")
    result = []
    for w in windows:
        if not np.isfinite(w).all():
            result.append(np.nan)
        else:
            values, counts = np.unique(w, return_counts=True)
            winners = values[counts == counts.max()]
            result.append(w[-1] if w[-1] in winners else winners[0])
    return np.asarray(result)


def clustering_quality(z, labels=None, n_clusters=3):
    z = np.asarray(z)
    source = "true_regime" if labels is not None else "kmeans"
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("invalid embeddings")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (len(z),):
            raise ValueError("labels must match embeddings")
        valid = np.isfinite(labels)
        z, labels = z[valid], labels[valid]
    if len(z) > 2000:
        idx = np.random.default_rng(0).choice(len(z), 2000, replace=False)
        z, labels = z[idx], None if labels is None else labels[idx]
    if labels is None and len(z)>n_clusters and len(np.unique(z, axis=0))>=n_clusters:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(z)
    groups = 0 if labels is None else len(np.unique(labels))
    if not 2 <= groups < len(z):
        return dict(silhouette=None, label_source=source, n_clusters=groups,
                    note="silhouette requires 2..N-1 nonempty groups; no fallback to another task")
    return dict(silhouette=float(silhouette_score(z, labels)), label_source=source,
                n_clusters=groups, note=None)


def anomaly_scores(x, x_hat, kl_per_window=None, lam=0.):
    if x.shape != x_hat.shape or x.ndim != 3 or lam < 0:
        raise ValueError("invalid reconstruction or KL weight")
    score = np.mean((x-x_hat)**2, axis=(1, 2))
    if kl_per_window is not None:
        if np.asarray(kl_per_window).shape != score.shape:
            raise ValueError("KL shape mismatch")
        score = score + lam*np.asarray(kl_per_window)
    if not np.isfinite(score).all():
        raise ValueError("nonfinite anomaly scores")
    return score


def calibrate_threshold(validation_scores, *, quantile=.95, validation_labels=None):
    scores = np.asarray(validation_scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 < quantile < 1:
        raise ValueError("invalid calibration scores/quantile")
    if validation_labels is None:
        return float(np.quantile(scores, quantile))
    labels = np.asarray(validation_labels)
    if labels.shape != scores.shape or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) < 2:
        raise ValueError("supervised calibration needs both classes in validation")
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))
    # Include an all-positive rule; ties prefer the more conservative threshold.
    thresholds = np.r_[np.nextafter(scores.min(), -np.inf), thresholds]
    values = [f1_score(labels, scores>t, zero_division=0) for t in thresholds]
    return float(thresholds[np.flatnonzero(values == np.max(values))[-1]])


def anomaly_detection_f1(x, x_hat, is_anomaly_window, *, threshold, kl_per_window=None, lam=0.):
    if not np.isfinite(threshold):
        raise ValueError("fixed finite validation-calibrated threshold required")
    scores = anomaly_scores(x, x_hat, kl_per_window, lam)
    labels = np.asarray(is_anomaly_window)
    report = binary_report(labels, (scores>threshold).astype(int), scores)
    return {**report, "threshold": float(threshold), "anomaly_rate_true": float(labels.mean()),
            "predicted_positive_rate": float((scores>threshold).mean())}


def window_anomaly_labels(is_anomaly, window_len, stride, n_windows):
    return np.asarray([int(w.any()) for w in _label_windows(is_anomaly, window_len, stride, n_windows)])


def event_report(labels, predictions):
    """Event recall counts contiguous labelled episodes, not overlapping windows."""
    labels, predictions = np.asarray(labels, bool), np.asarray(predictions, bool)
    if labels.shape != predictions.shape:
        raise ValueError("event arrays mismatch")
    boundaries = np.flatnonzero(np.diff(np.r_[False, labels, False].astype(int)))
    events = list(zip(boundaries[::2], boundaries[1::2]))
    detected = sum(predictions[a:b].any() for a,b in events)
    return dict(n_events=len(events), detected_events=int(detected),
                event_recall=float(detected/len(events)) if events else None,
                false_positive_points=int((predictions & ~labels).sum()))
