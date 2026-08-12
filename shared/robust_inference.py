"""Robust regression and block-bootstrap inference for football backtests."""
from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class OLSResult:
    beta: np.ndarray; covariance: np.ndarray; se: np.ndarray; t_values: np.ndarray
    residuals: np.ndarray; fitted: np.ndarray; r_squared: float; residual_sd: float
    n: int; dof_resid: int; covariance_type: str; n_clusters: int | None = None

@dataclass(frozen=True)
class BootstrapResult:
    point: float; low: float; high: float; replicates: np.ndarray; n_blocks: int

def _clusters(groups: Iterable[object], n: int):
    raw=list(groups)
    if len(raw)!=n: raise ValueError("groups and response must have the same number of rows")
    positions={}
    for i,g in enumerate(raw):
        key=tuple(g.tolist()) if isinstance(g,np.ndarray) else (tuple(g) if isinstance(g,list) else g)
        try: positions.setdefault(key,[]).append(i)
        except TypeError as exc: raise ValueError("each cluster label must be hashable") from exc
    return [np.asarray(v,dtype=int) for v in positions.values()]

def fit_ols(response, predictors, *, covariance="hc3", groups=None, add_intercept=True):
    y=np.asarray(response,dtype=float); x=np.asarray(predictors,dtype=float)
    if y.ndim!=1 or not np.isfinite(y).all(): raise ValueError("response must be one-dimensional and finite")
    if x.ndim==1: x=x.reshape(-1,1)
    if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all(): raise ValueError("predictors must be finite and match the response")
    if add_intercept: x=np.column_stack([np.ones(len(y)),x])
    n,k=x.shape
    if n<=k or np.linalg.matrix_rank(x)<k: raise ValueError("OLS requires a full-rank design with n > k")
    inv=np.linalg.inv(x.T@x); beta=inv@x.T@y; fitted=x@beta; resid=y-fitted; dof=n-k
    resid_sd=float(np.sqrt((resid@resid)/dof)); key=covariance.lower(); n_clusters=None
    if key=="iid": cov=inv*resid_sd**2
    elif key=="hc3":
        leverage=np.einsum("ij,jk,ik->i",x,inv,x); adj=resid/np.maximum(1-leverage,np.finfo(float).eps)
        cov=inv@(x.T@(x*np.square(adj)[:,None]))@inv
    elif key=="cluster":
        if groups is None: raise ValueError("groups are required for clustered covariance")
        blocks=_clusters(groups,n); n_clusters=len(blocks)
        if n_clusters<2: raise ValueError("clustered covariance requires at least two clusters")
        meat=np.zeros((k,k))
        for idx in blocks:
            score=x[idx].T@resid[idx]; meat+=np.outer(score,score)
        cov=(n_clusters/(n_clusters-1))*((n-1)/dof)*inv@meat@inv
    else: raise ValueError(f"unknown covariance estimator: {covariance!r}")
    diag=np.diag(cov); diag=np.where(np.abs(diag)<1e-15,0.0,diag)
    se=np.sqrt(np.where(diag>=0,diag,np.nan)); t=np.divide(beta,se,out=np.zeros_like(beta),where=se>0)
    sst=float(np.square(y-y.mean()).sum()); ssr=float(resid@resid)
    return OLSResult(beta,cov,se,t,resid,fitted,1-ssr/sst if sst>0 else 0.0,resid_sd,n,dof,key,n_clusters)

def season_block_bootstrap(values, seasons, *, statistic: Callable=np.mean, n_boot=10000, seed=20260810, confidence=.90):
    sample=np.asarray(values,dtype=float)
    if sample.ndim!=1 or not np.isfinite(sample).all(): raise ValueError("values must be one-dimensional and finite")
    if n_boot<=0 or not 0<confidence<1: raise ValueError("invalid bootstrap settings")
    blocks=_clusters(seasons,len(sample)); rng=np.random.default_rng(seed); reps=np.empty(n_boot)
    if not blocks: raise ValueError("at least one season block is required")
    for i in range(n_boot):
        selected=rng.integers(0,len(blocks),size=len(blocks)); idx=np.concatenate([blocks[j] for j in selected]); reps[i]=float(statistic(sample[idx]))
    tail=(1-confidence)/2
    return BootstrapResult(float(statistic(sample)),float(np.quantile(reps,tail)),float(np.quantile(reps,1-tail)),reps,len(blocks))

__all__=["BootstrapResult","OLSResult","fit_ols","season_block_bootstrap"]
