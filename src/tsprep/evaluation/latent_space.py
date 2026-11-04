"""Descriptive latent diagnostics; variance ratio is not information content."""
import numpy as np
from sklearn.decomposition import PCA


def information_collapse_ratio(z, x):
    """Legacy API: scale-dependent variance ratio, NOT an information metric."""
    if len(z) < 2 or len(z) != len(x):
        raise ValueError("at least two aligned samples required")
    return float(np.var(z, axis=0, ddof=1).sum() /
                 (np.var(x.reshape(len(x), -1), axis=0, ddof=1).sum()+1e-12))


def latent_diagnostics(z):
    if len(z) < 2:
        return dict(effective_rank=None, active_units=None)
    variances = np.var(z, axis=0, ddof=1)
    eigen = np.linalg.eigvalsh(np.atleast_2d(np.cov(z, rowvar=False)))
    eigen = np.maximum(eigen, 0)
    p = eigen/eigen.sum() if eigen.sum()>0 else eigen
    positive = p>0
    return dict(effective_rank=float(np.exp(-np.sum(p[positive]*np.log(p[positive])))) if positive.any() else 0.,
                active_units=int((variances>.01).sum()), active_units_variance_threshold=.01,
                coordinate_variances=variances.tolist())


def pca_project(z, n_components=2):
    if len(z) < 2 or z.shape[1] < n_components:
        raise ValueError("insufficient samples/features for requested PCA projection")
    return PCA(n_components=min(n_components, len(z)), random_state=0).fit_transform(z)


def tsne_project(z, n_components=2, perplexity=30.):
    from sklearn.manifold import TSNE
    if len(z) < 3 or z.shape[1] < n_components or perplexity <= 0:
        raise ValueError("insufficient samples/features or invalid perplexity")
    return TSNE(n_components=n_components, perplexity=min(perplexity, len(z)-1),
                random_state=0, init="pca").fit_transform(z)
