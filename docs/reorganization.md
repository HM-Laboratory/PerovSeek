# Repository organization

The package and demonstration were reorganized from source commit
`d1a24245425614bb270606143cff3c84a1db5d76` in an isolated checkout.

| Location | Purpose |
| --- | --- |
| `perovseek/bayesian.py` | Nine-component PCE optimization and input validation |
| `perovseek/spectra.py` | Spectral preprocessing, trained PCE inference and plots |
| `perovseek/models/` | Original spectral architecture with package-relative imports |
| `perovseek/training.py` | Offline SSE/AdamW training and portable checkpoints |
| `notebooks/PerovSeek_demo.ipynb` | Single demonstration entry point |
| `scripts/train_spectral_model.py` | Command-line training entry point |
| `checkpoints/spectral_pce_state.pt` | Trained PCE regressor from the validated 100-epoch run |
| `checkpoints/spectral_encoder.pt` | Original encoder weights for offline fine-tuning |
| `data/formulations.xlsx` | Prepared nine-component measurements and search bounds |
| `data/formulations_sources.csv` | Row-level source mapping for the prepared measurements |
| `data/source/` | Unmodified original formulation and preview workbooks |
| `data/spectra/` | Unmodified original spectral workbooks |
| `legacy/` | Archived research notebooks and utilities |
| `outputs/` | Local generated results, excluded from Git |

Exact source-to-destination paths and original SHA256 values are recorded in
`file_moves.json`. Workbooks and the encoder retain their original bytes.
`model.py` becomes `perovseek/models/mae.py`; its transformer import changes
to a relative import, while the model layers and forward calculations are
preserved. `trans.py` becomes `perovseek/models/transformer.py` unchanged.

The training loop retains the original summed squared error, AdamW steps and
cosine learning-rate update order. It saves the best validation state directly
as a `state_dict`, instead of saving full Python model pickles. The bundled
PCE checkpoint is copied unchanged from the validated previous run.

The archived notebooks document earlier research code and retain their old
path references; they are not the supported demonstration entry point. Generated
Python bytecode is removed from tracked source and ignored on future runs.
