from scipy.optimize import minimize
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def mp_density(lam, gamma):
    """MP probability density function."""
    lam_plus  = (1 + np.sqrt(gamma))**2
    lam_minus = (1 - np.sqrt(gamma))**2
    mask = (lam >= lam_minus) & (lam <= lam_plus)
    density = np.zeros_like(lam)
    density[mask] = np.sqrt((lam_plus - lam[mask]) * (lam[mask] - lam_minus)) \
                    / (2 * np.pi * gamma * lam[mask])
    return density

        
def eigen_decomp(returns, method='factor'):
    T, N = returns.shape
    corr_mat = np.corrcoef(returns, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    lam_plus = (1 + np.sqrt(N / T))**2
    sig_mask = eigenvalues > lam_plus

    if method == 'flat':                               # your original: average the noise eigenvalues
        eigenvalues_clean = eigenvalues.copy()
        eigenvalues_clean[~sig_mask] = eigenvalues[~sig_mask].mean()
        C_clean = eigenvectors @ np.diag(eigenvalues_clean) @ eigenvectors.T
        d = np.sqrt(np.diag(C_clean))
        return C_clean / np.outer(d, d)

    V = eigenvectors[:, sig_mask]                      # 'factor': keep signal, noise -> diagonal
    C_clean = (V * eigenvalues[sig_mask]) @ V.T
    np.fill_diagonal(C_clean, 1.0)
    return C_clean

def rmt_cov(R):
    sd = R.std().values * np.sqrt(252)
    return eigen_decomp(R) * np.outer(sd, sd)

def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(squareform(dist.values, checks=False), 'single')
    # link = sch.ward(squareform(dist, checks=False))
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()

def correlDist(corr):
    dist = ((1-corr)/2.0)**0.5
    return dist

def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])  # item 2
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()

def getIVP(cov, **kargs):
    ivp = 1.0/np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov,cItems):
    cov_ = cov.loc[cItems,cItems]
    w_ = getIVP(cov_).reshape(-1,1)
    cVar = np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1.0, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems
                  for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                  if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]      # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha
            w[cItems1] *= 1 - alpha
    return w
    
# --- weighting methods: take a window of returns (DataFrame), return weights summing to 1 ---
def equal_weight(R):
    return np.full(R.shape[1], 1 / R.shape[1])

def trailing_growth_weight(R):
    g = (1 + R).prod()                       # growth of each strategy over the window
    return (g / g.sum()).values

def inverse_vol(R):
    iv = 1 / R.std().values
    return iv / iv.sum()

def hrp_weights(R, rmt=False):
    if rmt:
        C  = eigen_decomp(R)
        sd = R.std().values
        corr = pd.DataFrame(C, R.columns, R.columns)
        cov  = pd.DataFrame(C * np.outer(sd, sd), R.columns, R.columns)
    else:
        cov, corr = R.cov(), R.corr()
    dist = correlDist(corr)
    dist_vals = dist.values.copy()
    np.fill_diagonal(dist_vals, 0.0)
    link = sch.linkage(squareform(dist_vals, checks=False), 'single')

    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    return getRecBipart(cov, sortIx)

def _cov(R, shrink=0.0, rmt=False):
    if rmt:
        S = rmt_cov(R)      # or your eigen_decomp, if it returns a cov
    else:
        S = R.cov().values * 252
    return (1 - shrink) * S + shrink * np.diag(np.diag(S))

def _solve(obj, n, w_max):
    w0 = np.full(n, 1 / n)
    res = minimize(obj, w0, method='SLSQP',
                   bounds=[(0, w_max)] * n,
                   constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},),
                   options={'ftol': 1e-12, 'maxiter': 500})
    w = np.clip(res.x, 0, None) if res.success else w0       # fall back to EW if it fails
    return w / w.sum()

def max_sharpe(R, w_max=1.0, shrink=0.0, rmt=False):
    mu, S = R.mean().values * 252, _cov(R, shrink, rmt)
    return _solve(lambda w: -(w @ mu) / np.sqrt(w @ S @ w), len(mu), w_max)

def min_variance(R, w_max=1.0, shrink=0.0, rmt=False):
    S = _cov(R, shrink, rmt)
    return _solve(lambda w: w @ S @ w, S.shape[0], w_max)

def risk_parity(R, rmt=False):
    S = _cov(R, 0.0, rmt)
    n = S.shape[0]
    f = lambda y: 0.5 * y @ S @ y - np.log(y).sum() / n        # Spinu's convex form
    g = lambda y: S @ y - 1 / (n * y)
    y = minimize(f, np.full(n, 1 / n), jac=g, method='L-BFGS-B',
                 bounds=[(1e-10, None)] * n).x
    return y / y.sum()

def ew_filtered(R, window=126, min_sr=0.0):
    X = R.iloc[-window:]
    sr = X.mean() / X.std() * np.sqrt(252)
    keep = (sr > min_sr).to_numpy(copy=True)                   # writable under pandas 3
    if not keep.any():
        keep[:] = True                                          # nothing passes: fall back to EW
    return keep / keep.sum()