"""
Downstream-задачи оценки качества латентного представления
  1) Прогнозирование: линейная регрессия z_t -> y_{t+H} (простая,
     "слабая" голова - если ДАЖЕ линейная модель на эмбеддингах
     показывает низкую ошибку, значит эмбеддинг несёт релевантную
     информацию о будущей динамике).
  2) Кластеризация: silhouette score эмбеддингов относительно скрытых
     рыночных режимов (в реальных данных - относительно kmeans-меток,
     если истинные режимы неизвестны).
  3) Обнаружение аномалий: композитный скор s(x) = recon_error + λ·KL,
     F1 относительно внесённых точечных аномалий.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              roc_auc_score, silhouette_score)


def forecast_via_linear_probe(
    z_train: np.ndarray, y_train: np.ndarray,
    z_test: np.ndarray, y_test: np.ndarray,
) -> dict:
    """
    Линейный "зонд" (linear probe) на замороженных эмбеддингах:
    обучаем LinearRegression(z -> y) и меряем MAE на тесте.
    """
    reg = LinearRegression().fit(z_train, y_train)
    y_pred = reg.predict(z_test)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(reg.score(z_test, y_test)),
    }


def classify_via_linear_probe(
    z_train: np.ndarray, y_train: np.ndarray,
    z_test: np.ndarray, y_test: np.ndarray,
    class_weight: str | None = "balanced",
) -> dict:
    """
    Линейный классификатор на замороженных эмбеддингах для предсказания классов,
    например рецессий NBER. По умолчанию используется балансировка классов.
    Для расчёта метрик в train и test нужны оба класса; иначе возвращаются None и пояснение в note.
    """
    classes_train = np.unique(y_train)
    classes_test = np.unique(y_test)
    if len(classes_train) < 2:
        return {"accuracy": None, "f1": None, "roc_auc": None,
                "note": f"only one class in train ({classes_train}) - probe not fitted"}
    if len(classes_test) < 2:
        # обучить можно, но ROC-AUC на одном классе не определён
        clf = LogisticRegression(max_iter=1000, class_weight=class_weight).fit(z_train, y_train)
        y_pred = clf.predict(z_test)
        return {"accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": None,
                "note": f"only one class in test ({classes_test}) - roc_auc undefined"}

    clf = LogisticRegression(max_iter=1000, class_weight=class_weight).fit(z_train, y_train)
    y_pred = clf.predict(z_test)
    y_proba = clf.predict_proba(z_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "note": None,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }


def window_class_labels(labels: np.ndarray, window_len: int, stride: int, n_windows: int,
                         rule: str = "last") -> np.ndarray:
    """
    Сворачивает поточечную разметку в метку на уровне окна

    rule='last'     - метка последнего шага окна (соответствует тому,
                       что энкодер видит "к моменту принятия решения");
    rule='majority'  - метка большинства точек внутри окна.
    """
    out = np.zeros(n_windows, dtype=int)
    for k, start in enumerate(range(0, n_windows * stride, stride)):
        window_labels = labels[start:start + window_len]
        if rule == "last":
            out[k] = window_labels[-1]
        elif rule == "majority":
            out[k] = int(np.round(window_labels.mean()))
        else:
            raise ValueError(f"Unknown rule: {rule}")
    return out


def clustering_quality(z: np.ndarray, labels: np.ndarray | None = None, n_clusters: int = 3) -> dict:
    """
    Silhouette score латентных представлений. Если истинные метки режима
    заданы (синтетика) - используем их напрямую; иначе кластеризуем
    KMeans'ом и считаем silhouette по полученным кластерам.
    """
    if z.shape[0] > 5000:
        # для скорости считаем silhouette на случайной подвыборке
        rng = np.random.default_rng(0)
        idx = rng.choice(z.shape[0], size=5000, replace=False)
        z_sub = z[idx]
        labels_sub = labels[idx] if labels is not None else None
    else:
        z_sub, labels_sub = z, labels

    if labels_sub is not None and len(np.unique(labels_sub)) > 1:
        score = silhouette_score(z_sub, labels_sub)
        return {"silhouette": float(score), "label_source": "true_regime"}

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(z_sub)
    score = silhouette_score(z_sub, km.labels_)
    return {"silhouette": float(score), "label_source": "kmeans"}


def anomaly_detection_f1(
    x: np.ndarray, x_hat: np.ndarray, is_anomaly_window: np.ndarray,
    z: np.ndarray | None = None, kl_per_window: np.ndarray | None = None,
    lam: float = 0.1,
) -> dict:
    """
    Скор аномальности окна: 
    s(x) = ||X - X_hat||_2 (+ lambda * KL, если задан).
    Порог выбирается по квантилю этих же скоров без использования тестовых меток;
    метки применяются только для расчёта итоговой F1.
    """
    recon_err = np.mean((x - x_hat) ** 2, axis=(1, 2))
    score = recon_err if kl_per_window is None else recon_err + lam * kl_per_window

    best_f1, best_thr = 0.0, None
    for q in np.linspace(0.80, 0.999, 60):
        thr = np.quantile(score, q)
        pred = (score > thr).astype(int)
        f1 = f1_score(is_anomaly_window, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return {"f1": float(best_f1), "threshold": float(best_thr) if best_thr is not None else None,
            "anomaly_rate_true": float(is_anomaly_window.mean())}


def window_anomaly_labels(is_anomaly: np.ndarray, window_len: int, stride: int, n_windows: int) -> np.ndarray:
    """Окно считается аномальным, если внутри него есть хотя бы одна
    внесённая точечная аномалия"""
    labels = np.zeros(n_windows, dtype=int)
    for k, start in enumerate(range(0, n_windows * stride, stride)):
        if is_anomaly[start:start + window_len].any():
            labels[k] = 1
    return labels