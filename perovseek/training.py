"""Offline fine-tuning of the spectral PCE regressor."""
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import FineTuneModel
from .models.factory import create_encoder
from .spectra import PREPROCESSING, load_spectra


def train_spectral_model(
    data, encoder_checkpoint, output, *, epochs=100, seed=42,
    train_count=500, test_count=1500, batch_size=32,
    lr=1e-3, min_lr=1e-4, weight_decay=1e-4,
    device="cpu", num_threads=2, overwrite=False,
):
    """Run the original SSE/AdamW/cosine loop and save the best state_dict.

    The learning-rate update follows the original utility.fine_train order:
    optimizer step, then cosine update at epoch + 1 + batch_index / batches.
    Only checkpoint serialization changes; full model pickles are not written.
    """
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint exists: {output}. Use --overwrite explicitly.")
    for name, value in dict(epochs=epochs, train_count=train_count,
                            test_count=test_count, batch_size=batch_size,
                            num_threads=num_threads).items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not (0 < min_lr <= lr) or not np.isfinite([lr, min_lr, weight_decay]).all():
        raise ValueError("Learning rates must satisfy 0 < min_lr <= lr.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative.")
    spectra = load_spectra(data)
    if spectra.measured_pce is None:
        raise ValueError("Training requires PCE labels in the Param sheet.")
    if train_count + test_count >= len(spectra):
        raise ValueError("The split must leave at least one validation sample.")
    np.random.seed(seed)
    indices = np.arange(len(spectra))
    np.random.shuffle(indices)
    train_rows = indices[:train_count]
    validation_rows = indices[train_count:-test_count]
    test_rows = indices[-test_count:]
    inputs = torch.tensor(spectra.inputs, dtype=torch.float32)
    labels = torch.tensor(spectra.measured_pce[:, None] / 25.0, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(inputs[train_rows], labels[train_rows]),
        batch_size=batch_size, shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(inputs[validation_rows], labels[validation_rows]),
        batch_size=batch_size, shuffle=False,
    )
    torch.manual_seed(seed)
    encoder = create_encoder(device)
    encoder_weights = torch.load(
        encoder_checkpoint, weights_only=True, map_location=device
    )
    encoder.load_state_dict(encoder_weights, strict=True)
    architecture = dict(hidden_dim=[18], n=10, embed_dim=36, dropout_rate=None)
    model = FineTuneModel(encoder, **architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    settings = dict(batch_size=batch_size, lr=lr, min_lr=min_lr, epochs=epochs,
                    weight_decay=weight_decay, warmup_epochs=0, device=device)
    metadata = {
        "dataset": spectra.source_path.name,
        "label": "PCE#rs1",
        "data_sha256": spectra.source_sha256,
        "pretrained_sha256": hashlib.sha256(Path(encoder_checkpoint).read_bytes()).hexdigest(),
        "architecture": architecture,
        "training": settings,
        "seed": seed,
        "splits": {"train": train_rows.tolist(),
                   "validation": validation_rows.tolist(), "test": test_rows.tolist()},
        "preprocessing": PREPROCESSING.copy(),
    }
    history = {"epoch": [], "train_loss": [], "validation_loss": [], "learning_rate": []}
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(num_threads)
        for epoch in range(epochs):
            model.train()
            training_loss = 0.0
            for batch_index, (batch, target) in enumerate(train_loader):
                optimizer.zero_grad()
                prediction, _ = model(batch.to(device))
                loss = torch.sum((prediction - target.to(device)) ** 2)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                current_epoch = epoch + 1 + batch_index / len(train_loader)
                current_lr = min_lr + (lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * current_epoch / epochs)
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch, target in validation_loader:
                    prediction, _ = model(batch.to(device))
                    validation_loss += torch.sum((prediction - target.to(device)) ** 2).item()
            training_loss /= len(train_loader)
            validation_loss /= len(validation_loader)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(training_loss)
            history["validation_loss"].append(validation_loss)
            history["learning_rate"].append(current_lr)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            print(f"Epoch {epoch + 1:3d}/{epochs}: "
                  f"train={training_loss:.6f} validation={validation_loss:.6f}",
                  flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    payload = {"format_version": 1, "state_dict": best_state,
               "best_epoch": best_epoch, "metadata": metadata, "history": history}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    pd.DataFrame(history).to_csv(output.with_suffix(".history.csv"), index=False)
    return {"checkpoint": str(output), "best_epoch": best_epoch,
            "validation_loss": best_loss, "epochs": epochs}
