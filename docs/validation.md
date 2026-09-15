# Validation

Run the workflow checks from the repository root after installation:

```bash
python -m unittest discover -s tests -v
```

The checks cover feasible Excel bounds, missing measured PCE, the solvent total, spectral sample identifiers and inference with the supplied checkpoint.

## Verify the demonstration notebook

Open `notebooks/PerovSeek_demo.ipynb`, restart its Python 3.12 kernel and choose **Run All Cells**. Verify all eight code cells complete, then check:

1. The input contains 126 measured formulations. The six exported candidates satisfy the Excel bounds and DMF + NFM + EA = 1 within a tolerance of `1e-6`.
2. `load_spectra(..., label=None)` loads 2,268 spectral records, and `subset="all"` produces the same number of predictions. The notebook displays absorption and PL spectra and a ranking of eight records.
3. `spectral_predictions.csv` and `ranked_spectra.csv` preserve `sample_id` and `source_row`. Prediction results contain no measured-PCE or error columns.
4. `device_results.csv` contains eight rows with `sample_id`, `source_row` and finite `PCE`. Each value equals the original `Param.iloc[source_row]["PCE#rs1"]`, and its identifier matches `Param.iloc[source_row]["Unnamed: 0"]`. Repeated sample identifiers must remain separate records.
5. The final displayed table contains `Sample` and `PCE (%)`. The formulation workbook still contains its original 126 measured records.

All result files are written under `outputs/`. This checklist describes the eight-cell demonstration's acceptance checks; it is separate from the model reference evaluation below.

The current eight-cell notebook was executed in a fresh local kernel with no errors. It produced 2,268 predictions, eight ranked records and eight device-PCE records. Each device value was checked against its original `Param` row, with the sample identifier verified. All five library checks passed, and the source workbooks remained unchanged. The committed notebook includes these executed outputs for direct preview.

## Model reference evaluation

The library supports an optional evaluation with `load_spectra(path)` and `predict(..., subset="test")`. The saved 1,500-record test split is restricted to the exact source-workbook hash. `prediction_metrics` and `plot_predictions` remain available for this evaluation; the demonstration notebook uses all-record inference and does not calculate test metrics or display a measured-versus-predicted plot.

The supplied checkpoint's 1,500 test predictions reproduce the reference values within 0.000002 PCE percentage points. MAE is 2.265760 percentage points, RMSE is 3.552679 percentage points, and R² is 0.611534. These results describe the example split specified in [data_sources.md](data_sources.md), not the all-record ranking.

The separate training command was run for 100 epochs in a CPU reproduction check. It selected epoch 47 and reproduced every tensor of the supplied PCE checkpoint, as well as the training history. Retraining is optional and is not part of the demonstration notebook.

Original workbooks and encoder weights were checked against the hashes in [file_moves.json](file_moves.json). The original model computations are preserved with package-relative imports. Inference loads a state dictionary with `weights_only=True` and performs no network requests.
