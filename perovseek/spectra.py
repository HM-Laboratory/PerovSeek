"""Read spectra and infer PCE with the trained portable regression checkpoint."""
from dataclasses import dataclass
import hashlib
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .models import FineTuneModel
from .models.factory import create_encoder

PREPROCESSING = {
    "abs_divisor": 4,
    "pl_top_divisor": 120000,
    "pl_bottom_divisor": 150000,
    "wavelength_offset_nm": 400,
    "pce_scale": 25.0,
}


@dataclass
class SpectralData:
    """Model input and its source information, in workbook column order."""

    inputs: np.ndarray
    sample_ids: np.ndarray
    measured_pce: np.ndarray | None
    abs_wavelength: np.ndarray
    pl_wavelength: np.ndarray
    source_path: Path
    source_sha256: str

    def __len__(self):
        return len(self.inputs)


def load_spectra(path, *, label="PCE#rs1"):
    """Read Abs, PL_top and PL_bottom sheets; Param labels are optional.

    Each spectral sheet has wavelength in its first column. The absorption
    input uses the 300 rows at 402–1000 nm; both PL inputs use the first 66
    wavelength rows. Signals and wavelength offsets follow the training scale.
    """
    path = Path(path)
    tables = pd.read_excel(path, sheet_name=None)
    required = {"Abs", "PL_top", "PL_bottom"}
    missing = required - set(tables)
    if missing:
        raise ValueError(f"Missing spectral sheets: {sorted(missing)}")
    absorption, top, bottom = (tables[k] for k in ("Abs", "PL_top", "PL_bottom"))
    count = absorption.shape[1] - 1
    if count < 1 or top.shape[1] - 1 != count or bottom.shape[1] - 1 != count:
        raise ValueError("All spectral sheets must contain the same samples.")
    if list(absorption.columns[1:]) != list(top.columns[1:]) or (
        list(top.columns[1:]) != list(bottom.columns[1:])
    ):
        raise ValueError("Spectral sample columns must match in name and order.")
    abs_wave = absorption.iloc[1:, 0].to_numpy(float)
    pl_wave = top.iloc[:66, 0].to_numpy(float)
    bottom_wave = bottom.iloc[:66, 0].to_numpy(float)
    if not np.array_equal(abs_wave, np.arange(402, 1002, 2)):
        raise ValueError("Absorption wavelengths must cover 402–1000 nm in 2 nm steps.")
    if len(pl_wave) != 66 or not np.array_equal(pl_wave, bottom_wave):
        raise ValueError("Top and bottom PL require the same 66 wavelengths.")
    if not np.isfinite(pl_wave).all() or not np.all(np.diff(pl_wave) > 0):
        raise ValueError("PL wavelengths must be finite and strictly increasing.")
    signal = np.concatenate([
        absorption.iloc[1:, 1:].to_numpy(float).T / 4,
        top.iloc[:66, 1:].to_numpy(float).T / 120000,
        bottom.iloc[:66, 1:].to_numpy(float).T / 150000,
    ], axis=1)
    wavelength = np.concatenate([abs_wave, pl_wave, pl_wave]) - 400
    inputs = np.stack([signal, np.broadcast_to(wavelength, signal.shape)], axis=1)
    if inputs.shape != (count, 2, 432) or not np.isfinite(inputs).all():
        raise ValueError("The spectral input must be finite with shape (samples, 2, 432).")
    measured = None
    sample_ids = np.asarray([str(value) for value in absorption.columns[1:]])
    if "Param" in tables:
        parameters = tables["Param"]
        if len(parameters) != count:
            raise ValueError("Param rows must align with the spectral sample columns.")
        if label in parameters:
            measured = parameters[label].to_numpy(float)
            if np.isnan(measured).all():
                measured = None
            elif not np.isfinite(measured).all():
                raise ValueError("Provided PCE labels must be finite.")
        if "sample_id" in parameters:
            sample_ids = parameters["sample_id"].astype(str).to_numpy()
        elif str(parameters.columns[0]).startswith("Unnamed"):
            sample_ids = parameters.iloc[:, 0].astype(str).to_numpy()
    return SpectralData(
        inputs=inputs, sample_ids=sample_ids, measured_pce=measured,
        abs_wavelength=abs_wave, pl_wavelength=pl_wave,
        source_path=path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


class SpectralPredictor:
    """Inference with trained PCE regression weights; no training is performed."""

    def __init__(self, checkpoint, *, device="cpu", num_threads=2):
        if not isinstance(num_threads, Integral) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer.")
        self.device = torch.device(device)
        self.num_threads = num_threads
        self.checkpoint = Path(checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if not {"state_dict", "metadata", "history", "best_epoch"}.issubset(payload):
            raise ValueError("A trained PCE checkpoint is required, not encoder weights.")
        self.metadata = payload["metadata"]
        if self.metadata.get("preprocessing") != PREPROCESSING:
            raise ValueError("The checkpoint uses an incompatible spectral input scale.")
        self.history = pd.DataFrame(payload["history"])
        self.best_epoch = int(payload["best_epoch"])
        with torch.random.fork_rng(devices=[]):
            encoder = create_encoder()
            self.model = FineTuneModel(encoder, **self.metadata["architecture"])
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval()

    def predict(self, spectra, *, subset="all", batch_size=128):
        """Return unclipped PCE predictions, optionally for the saved test split.

        subset='test' is restricted to the exact training-source workbook hash.
        subset='all' permits new compatible spectra, without requiring labels.
        """
        if not isinstance(spectra, SpectralData):
            raise TypeError("Load a SpectralData object with load_spectra first.")
        if not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if subset == "all":
            rows = np.arange(len(spectra))
        elif subset == "test":
            if spectra.source_sha256 != self.metadata["data_sha256"]:
                raise ValueError("The saved test split belongs to a different workbook.")
            rows = np.asarray(self.metadata["splits"]["test"], dtype=int)
        else:
            raise ValueError("subset must be 'all' or 'test'.")
        if len(rows) == 0 or rows.min() < 0 or rows.max() >= len(spectra):
            raise ValueError("The selected sample rows are outside the spectral data.")
        inputs = torch.tensor(spectra.inputs[rows], dtype=torch.float32)
        if not torch.isfinite(inputs).all():
            raise ValueError("Model input contains nonfinite values.")
        batches = []
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self.num_threads)
            with torch.inference_mode():
                for batch in inputs.split(batch_size):
                    scaled, _ = self.model(batch.to(self.device))
                    batches.append(scaled.cpu().numpy().ravel() * 25.0)
        finally:
            torch.set_num_threads(previous_threads)
        prediction = np.concatenate(batches).astype(float)
        if not np.isfinite(prediction).all():
            raise RuntimeError("The model produced a nonfinite prediction.")
        result = pd.DataFrame({
            "sample_id": spectra.sample_ids[rows],
            "source_row": rows,
            "predicted_pce_percent": prediction,
        })
        if spectra.measured_pce is not None:
            result["measured_pce_percent"] = spectra.measured_pce[rows]
            result["error_percentage_points"] = (
                prediction - spectra.measured_pce[rows]
            )
        return result


def prediction_metrics(predictions):
    """Return errors in percentage points using every supplied prediction."""
    required = ["predicted_pce_percent", "measured_pce_percent"]
    if not set(required).issubset(predictions.columns):
        raise ValueError("Measured PCE is required to calculate prediction errors.")
    values = predictions[required].to_numpy(float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Metric inputs must be nonempty and finite.")
    predicted, measured = values.T
    error = predicted - measured
    variance = np.sum((measured - measured.mean()) ** 2)
    return {
        "test_count": len(values),
        "mae_pp": float(np.abs(error).mean()),
        "rmse_pp": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1 - np.sum(error ** 2) / variance) if variance > 0 else None,
        "negative_predictions": int(np.sum(predicted < 0)),
    }


def plot_spectra(spectra, *, sample=0):
    """Return a figure of absorption and PL signals on the model input scale."""
    if not 0 <= sample < len(spectra):
        raise ValueError("sample is outside the spectral data.")
    views = [
        (spectra.abs_wavelength, spectra.inputs[:, 0, :300], "Absorption"),
        (spectra.pl_wavelength, spectra.inputs[:, 0, 300:366], "PL, top excitation"),
        (spectra.pl_wavelength, spectra.inputs[:, 0, 366:], "PL, bottom excitation"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
    for ax, (wave, signal, title) in zip(axes, views):
        ax.plot(wave, signal[:12].T, color="#147d92", alpha=0.15)
        ax.plot(wave, signal[sample], color="#147d92", lw=2)
        ax.set(title=title, xlabel="Wavelength (nm)", ylabel="Scaled signal")
    plt.close(fig)
    return fig


def plot_predictions(predictions):
    """Return measured-vs-predicted PCE without hiding negative model outputs."""
    metrics = prediction_metrics(predictions)
    measured = predictions.measured_pce_percent.to_numpy()
    predicted = predictions.predicted_pce_percent.to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.4), layout="constrained")
    ax.scatter(measured, predicted, s=13, alpha=0.4, color="#147d92")
    xlim = (min(0, np.floor(measured.min()) - 1),
            max(25, np.ceil(measured.max()) + 1))
    ylim = (min(0, np.floor(predicted.min()) - 1),
            max(25, np.ceil(predicted.max()) + 1))
    ax.plot(xlim, xlim, "--", color="gray", lw=1)
    ax.set(xlabel="Measured PCE (%)", ylabel="Predicted PCE (%)",
           xlim=xlim, ylim=ylim, title="1.59 eV spectral model")
    label = f"n = {metrics['test_count']}\nMAE = {metrics['mae_pp']:.2f} pp"
    ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top")
    plt.close(fig)
    return fig
