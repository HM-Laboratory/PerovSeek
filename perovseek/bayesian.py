"""Nine-component PCE optimization with a Gaussian-process surrogate."""
from numbers import Integral

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

FEATURES = (
    "DMF", "NFM", "EA", "Me-4", "Py3", "4PADCB",
    "4-FBSA", "F3EABr", "SPFBS",
)


def _training_data(formulations, target):
    if not isinstance(formulations, pd.DataFrame):
        raise TypeError("formulations must be a pandas DataFrame.")
    if formulations.columns.duplicated().any():
        raise ValueError("Duplicate formulation column names are not allowed.")
    if target != "PCE":
        raise ValueError("The supported measured target is PCE (%).")
    missing = set(FEATURES + (target,)) - set(formulations.columns)
    if missing:
        raise ValueError(f"Missing formulation columns: {sorted(missing)}")
    try:
        data = formulations[list(FEATURES) + [target]].to_numpy(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Formulations and PCE must contain numbers.") from error
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("At least two complete, finite measured rows are required.")
    if (data < 0).any() or (data[:, -1] > 100).any():
        raise ValueError("Components must be nonnegative; PCE must be 0–100 (%).")
    if not np.allclose(data[:, :3].sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("DMF, NFM and EA must be volume fractions summing to 1.")
    return data[:, :-1], data[:, -1:]


def _search_bounds(bounds, train_x):
    if bounds is None:
        limits = np.stack([train_x.min(axis=0), train_x.max(axis=0)])
    else:
        if not isinstance(bounds, pd.DataFrame):
            raise TypeError("bounds must be a Component/Lower/Upper DataFrame.")
        if not {"Component", "Lower", "Upper"}.issubset(bounds.columns):
            raise ValueError("Bounds requires Component, Lower and Upper columns.")
        if bounds.Component.duplicated().any():
            raise ValueError("Each component must have exactly one bounds row.")
        if set(bounds.Component) != set(FEATURES):
            raise ValueError("Bounds must list exactly the nine component names.")
        limits = bounds.set_index("Component").loc[
            list(FEATURES), ["Lower", "Upper"]
        ].to_numpy(float).T
    if not np.isfinite(limits).all() or (limits < 0).any():
        raise ValueError("Bounds must be finite and nonnegative.")
    if not np.all(limits[1] > limits[0]):
        raise ValueError("Every search upper bound must exceed its lower bound.")
    if (limits[:, :3] > 1).any():
        raise ValueError("Solvent fraction bounds cannot exceed 1.")
    if limits[0, :3].sum() > 1 or limits[1, :3].sum() < 1:
        raise ValueError("The solvent bounds cannot satisfy a total fraction of 1.")
    return torch.tensor(limits, dtype=torch.double)


def recommend(
    formulations, *, target="PCE", bounds=None, batch_size=6,
    noise_sd=2.0, mc_samples=128, num_restarts=4, raw_samples=128,
    fit_maxiter=40, opt_maxiter=80, seed=42, num_threads=2,
):
    """Recommend unmeasured formulations within the supplied search bounds.

    Input columns are the nine names in FEATURES and measured PCE in percent.
    Solvents must already be normalized. Six other components retain the input
    table's numeric scale. When bounds is omitted, observed min/max are used.
    The only composition equality is DMF + NFM + EA = 1. Posterior estimates
    are returned in PCE percent, never as experimental measurements.
    """
    integers = dict(batch_size=batch_size, mc_samples=mc_samples,
                    num_restarts=num_restarts, raw_samples=raw_samples,
                    fit_maxiter=fit_maxiter, opt_maxiter=opt_maxiter,
                    num_threads=num_threads)
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1
           for v in integers.values()):
        raise ValueError("Batch and optimization counts must be positive integers.")
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError("noise_sd must be finite and positive.")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    x, y = _training_data(formulations, target)
    limits = _search_bounds(bounds, x)
    train_x = torch.tensor(x, dtype=torch.double)
    train_y = torch.tensor(y, dtype=torch.double)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(num_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            gp = SingleTaskGP(
                train_X=train_x, train_Y=train_y,
                train_Yvar=torch.full_like(train_y, noise_sd ** 2),
                covar_module=ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=9)),
                input_transform=Normalize(d=9, bounds=limits),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(
                mll, optimizer_kwargs={"options": {"maxiter": fit_maxiter}}
            )
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([mc_samples]), seed=seed
            )
            acquisition = qLogNoisyExpectedImprovement(
                model=gp, X_baseline=train_x, sampler=sampler
            )
            equality = [(torch.tensor([0, 1, 2]),
                         torch.ones(3, dtype=torch.double), 1.0)]
            candidates, value = optimize_acqf(
                acquisition, bounds=limits, q=batch_size,
                num_restarts=num_restarts, raw_samples=raw_samples,
                options={"batch_limit": 2, "maxiter": opt_maxiter},
                equality_constraints=equality,
            )
            with torch.no_grad():
                posterior = gp.posterior(candidates)
    finally:
        torch.set_num_threads(previous_threads)
    values = candidates.detach().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("Optimization returned nonfinite candidates.")
    if not np.allclose(values[:, :3].sum(1), 1, atol=1e-6, rtol=0):
        raise RuntimeError("Optimization violated the solvent-sum constraint.")
    if (values < limits[0].numpy() - 1e-6).any() or (
        values > limits[1].numpy() + 1e-6
    ).any():
        raise RuntimeError("Optimization returned candidates outside the bounds.")
    result = pd.DataFrame(values, columns=FEATURES)
    result.insert(0, "candidate_id", [f"D{i + 1:02d}" for i in range(batch_size)])
    result["PCE_pred"] = posterior.mean.numpy().ravel()
    result["PCE_std"] = posterior.variance.sqrt().numpy().ravel()
    result.attrs["target"] = target
    result.attrs["acquisition_value"] = float(value.detach())
    result.attrs["measured_records"] = len(formulations)
    return result
