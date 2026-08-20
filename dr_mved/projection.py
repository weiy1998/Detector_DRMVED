"""Domain-regularized lower-tail projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize


class ProjectionFailure(RuntimeError):
    """Raised when the prescribed scene-risk margin is not positive."""


def _solve_primal_socp(
    scatter: np.ndarray, deltas: np.ndarray, beta: float
) -> tuple[np.ndarray, float]:
    """Solve the paper's primal SOCP when optional CVXPY is installed."""

    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install cvxpy to use solver='cvxpy'.") from exc
    scene_count = deltas.shape[0]
    weight = cp.Variable(3)
    tau = cp.Variable()
    slack = cp.Variable(scene_count, nonneg=True)
    constraints = [
        cp.quad_form(weight, scatter) <= 1.0,
        slack >= tau - deltas @ weight,
    ]
    objective = cp.Maximize(tau - cp.sum(slack) / (beta * scene_count))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weight.value is None:
        raise ProjectionFailure(f"SOCP solver failed with status {problem.status}.")
    result = np.asarray(weight.value, dtype=float).reshape(3)
    value = float(problem.value)
    return result, value


@dataclass
class ProjectionFit:
    weight: np.ndarray
    scatter: np.ndarray
    within_scatter: np.ndarray
    drift_scatter: np.ndarray
    scene_deltas: np.ndarray
    mixture_weights: np.ndarray
    kappa: float
    beta: float
    solver: str = "capped-simplex-dual"

    def lower_tail_margin(self) -> float:
        margins = self.scene_deltas @ self.weight
        return float(np.min(self.mixture_weights @ margins))


def _covariance(features: np.ndarray, dimension: int) -> np.ndarray:
    if features.shape[0] < dimension + 1:
        raise ValueError(
            f"Each scene/class requires at least d+1={dimension + 1} samples."
        )
    covariance = np.cov(features, rowvar=False, ddof=1)
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    return covariance.reshape(dimension, dimension)


def _capped_mixture_weights(whitened_deltas: np.ndarray, beta: float) -> np.ndarray:
    """Minimize the squared norm over the capped simplex P_beta."""

    scene_count = whitened_deltas.shape[0]
    if not 1.0 / scene_count <= beta <= 1.0:
        raise ValueError(f"beta must lie in [1/S, 1] with S={scene_count}.")
    cap = 1.0 / (beta * scene_count)
    initial = np.full(scene_count, 1.0 / scene_count, dtype=float)
    if np.isclose(beta, 1.0):
        return initial

    def objective(weights: np.ndarray) -> float:
        mixture = weights @ whitened_deltas
        return 0.5 * float(mixture @ mixture)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, cap)] * scene_count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success or np.any(result.x < -1e-8) or abs(np.sum(result.x) - 1.0) > 1e-6:
        raise ProjectionFailure(f"Capped-mixture optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, cap)
    weights /= np.sum(weights)
    return weights


def fit_projection(
    features: np.ndarray,
    labels: Sequence[int],
    scene_ids: Sequence[Any],
    lambda_drift: float = 5.0,
    ridge: float = 0.05,
    beta: float = 0.5,
    solver: str = "auto",
    failure_tolerance: float = 1e-10,
) -> ProjectionFit:
    """Fit the signed projection from labelled scene-level CUT features.

    The implementation solves the dual capped-simplex convex problem, which is
    equivalent to the SOCP in the paper.  The returned direction is normalized
    so that ``w.T @ A @ w == 1`` up to floating-point error.
    """

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    scenes = np.asarray(scene_ids)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("features must have shape (n_samples, 3).")
    if not (len(X) == len(y) == len(scenes)):
        raise ValueError("features, labels, and scene_ids must have equal length.")
    if np.any(~np.isfinite(X)):
        raise ValueError("features contain non-finite values.")
    unique_scenes = list(dict.fromkeys(scenes.tolist()))
    if len(unique_scenes) < 1:
        raise ValueError("At least one training scene is required.")
    scene_deltas = []
    covariances = []
    null_centers = []
    for scene in unique_scenes:
        null = X[(scenes == scene) & (y == 0)]
        target = X[(scenes == scene) & (y == 1)]
        if null.shape[0] < 4 or target.shape[0] < 4:
            raise ValueError("Each scene must contain at least four CUTs per class.")
        null_center = np.mean(null, axis=0)
        target_center = np.mean(target, axis=0)
        scene_deltas.append(target_center - null_center)
        covariances.extend([_covariance(null, 3), _covariance(target, 3)])
        null_centers.append(null_center)
    deltas = np.asarray(scene_deltas, dtype=float)
    within = np.mean(np.asarray(covariances), axis=0)
    centers = np.asarray(null_centers, dtype=float)
    center = np.mean(centers, axis=0)
    centered = centers - center
    drift = (centered.T @ centered) / len(unique_scenes)
    scatter = within + float(lambda_drift) * drift + float(ridge) * np.eye(3)
    scatter = 0.5 * (scatter + scatter.T)

    try:
        chol = np.linalg.cholesky(scatter)
    except np.linalg.LinAlgError as exc:
        raise ProjectionFailure("Regularized scatter matrix is not positive definite.") from exc

    # For A=L L^T, A^{-1/2} is not needed explicitly: the dual geometry is
    # represented by the Mahalanobis norm m^T A^{-1}m.
    whitened = np.linalg.solve(chol, deltas.T).T
    weights = _capped_mixture_weights(whitened, float(beta))
    mixture = weights @ deltas
    inverse_mixture = np.linalg.solve(scatter, mixture)
    kappa_sq = float(mixture @ inverse_mixture)
    kappa = float(np.sqrt(max(kappa_sq, 0.0)))
    if kappa <= failure_tolerance:
        raise ProjectionFailure(
            "The fitted lower-tail margin is zero; the prescribed scene-risk "
            "requirement has no positive linear projection."
        )
    weight = inverse_mixture / kappa
    if float(weight @ mixture) < 0:
        weight = -weight
    used_solver = "capped-simplex-dual"
    if solver == "cvxpy":
        primal_weight, primal_value = _solve_primal_socp(scatter, deltas, float(beta))
        primal_norm = float(np.sqrt(max(primal_weight @ scatter @ primal_weight, 1e-15)))
        weight = primal_weight / primal_norm
        if float(weight @ mixture) < 0:
            weight = -weight
        used_solver = "cvxpy-socp"
    elif solver in {"auto", "dual", "capped-simplex-dual"}:
        used_solver = "capped-simplex-dual"
    else:
        raise ValueError(
            "solver must be 'auto', 'dual', 'capped-simplex-dual', or 'cvxpy'."
        )

    return ProjectionFit(
        weight=weight,
        scatter=scatter,
        within_scatter=within,
        drift_scatter=drift,
        scene_deltas=deltas,
        mixture_weights=weights,
        kappa=kappa,
        beta=float(beta),
        solver=used_solver,
    )
