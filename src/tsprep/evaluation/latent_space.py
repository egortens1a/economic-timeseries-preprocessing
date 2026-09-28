"""
Анализ латентного пространства для оценки структуры эмбеддингов и информационного коллапса.
Коэффициент mu - отношение изменчивости латентного представления к изменчивости входа:

mu = trace(Cov(z)) / trace(Cov(X_flat))

Значение mu -> 0 означает, что модель почти полностью схлопывает входную информацию
"""
import numpy as np
from sklearn.decomposition import PCA


def information_collapse_ratio(z: np.ndarray, x: np.ndarray) -> float:
    x_flat = x.reshape(x.shape[0], -1)
    cov_z = np.cov(z, rowvar=False)
    cov_x = np.cov(x_flat, rowvar=False)
    trace_z = np.trace(np.atleast_2d(cov_z))
    trace_x = np.trace(np.atleast_2d(cov_x))
    return float(trace_z / (trace_x + 1e-12))


def pca_project(z: np.ndarray, n_components: int = 2) -> np.ndarray:
    n_components = min(n_components, z.shape[1], z.shape[0])
    return PCA(n_components=n_components, random_state=0).fit_transform(z)


def tsne_project(z: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    perplexity = min(perplexity, max(5, z.shape[0] // 4))
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=0,
                init="pca").fit_transform(z)