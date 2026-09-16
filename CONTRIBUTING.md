# Contributing

## Development environment

Use Python 3.12 and install the project in a virtual environment:

```bash
python -m pip install -e '.[demo]'
python -m unittest discover -s tests -v
```

Keep the demonstration in `notebooks/PerovSeek_demo.ipynb`. Data loading, objectives and parameters should remain visible there; reusable optimization and model computations belong in `perovseek/`.

## Data and checkpoints

The supported inputs are `data/formulations.xlsx` and the workbooks in `data/spectra/`. Preserve their documented column names and units. Record changes to datasets, preprocessing or model weights in [Data and models](docs/data_sources.md), and update [checksums.sha256](docs/checksums.sha256) when these files change.

Do not commit generated outputs, virtual environments, notebook checkpoints or archived research files. Keep sample identifiers and source-row mappings when exporting predictions and device measurements.

## Before submitting a change

Run the tests and check any affected notebook cells. If data formats, model preprocessing or the public workflow change, restart the notebook kernel and run all cells as described in [Validation](docs/validation.md). Describe the behavior change and the checks you ran in the pull request.
