"""Checkpoint compatibility for the transparent PerovSeek demo notebook.

The original repository's utility.fine_train owns the scientific training loop.
This adapter only converts its locally created best-model pickle into a portable
state_dict checkpoint. Data, architecture, loss settings, split and optimizer
remain visible in PerovSeek_workflow.ipynb.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def fine_train_portable(
    model, optimizer, train_loader, validation_loader, params,
    checkpoint_path, metadata,
):
    """Call the original loop and retain a portable best-validation checkpoint.

    Only a pickle created inside this invocation is loaded with
    weights_only=False. The final checkpoint contains tensors and plain metadata
    and can be opened with torch.load(..., weights_only=True).
    """
    from utility import fine_train

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="training-", dir=checkpoint_path.parent) as temporary:
        train_loss, validation_loss, learning_rate, best_pickle = fine_train(
            model, optimizer, train_loader, validation_loader, params, temporary
        )
        if best_pickle is None:
            raise RuntimeError("The original training loop did not save a best model.")
        best_model = torch.load(best_pickle, map_location="cpu", weights_only=False)
        best_epoch = min(range(len(validation_loss)), key=validation_loss.__getitem__) + 1
        payload = {
            "format_version": 1,
            "state_dict": {name: value.detach().cpu()
                           for name, value in best_model.state_dict().items()},
            "best_epoch": best_epoch,
            "metadata": metadata,
            "history": {
                "epoch": list(range(1, len(train_loss) + 1)),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            },
        }
        torch.save(payload, checkpoint_path)
    return checkpoint_path
