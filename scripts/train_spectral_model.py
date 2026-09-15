"""Train a PCE checkpoint offline; the demonstration notebook only predicts."""
import argparse
import json

from perovseek.training import train_spectral_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/spectra/1.59eV_additive_data.xlsx")
    parser.add_argument("--encoder", default="checkpoints/spectral_encoder.pt")
    parser.add_argument("--output", default="outputs/training/spectral_pce_state.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = train_spectral_model(
        args.data, args.encoder, args.output,
        epochs=args.epochs, seed=args.seed,
        train_count=args.train_count, test_count=args.test_count,
        batch_size=args.batch_size, lr=args.lr, min_lr=args.min_lr,
        weight_decay=args.weight_decay, device=args.device,
        num_threads=args.num_threads, overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
