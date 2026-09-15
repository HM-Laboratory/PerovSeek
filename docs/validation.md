# Validation

Run the workflow checks from the repository root after installation:

```bash
python -m unittest discover -s tests -v
```

The five checks cover feasible Excel bounds, missing measured PCE, the solvent total, spectral sample identifiers and inference with the supplied checkpoint.

The nine demo cells were executed in a fresh kernel and through Jupyter's **Run All Cells** command. Both runs completed without errors. The supplied workbook produced six candidates within the bounds, with DMF + NFM + EA = 1.

The 1,500 saved test predictions reproduce the reference values within 0.000002 PCE percentage points. MAE is 2.265760 percentage points, RMSE is 3.552679 percentage points, and R² is 0.611534. These results describe the example split specified in [data_sources.md](data_sources.md).

The separate training command was also run for 100 epochs. It selected epoch 47 and reproduced every tensor of the supplied PCE checkpoint, as well as the training history, in the tested CPU environment.

Original workbooks and encoder weights were checked against the hashes in [file_moves.json](file_moves.json). The original model computations are preserved with package-relative imports. Inference loads a state dictionary with `weights_only=True` and performs no network requests.
