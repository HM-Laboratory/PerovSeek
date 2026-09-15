"""Checks for editable formulation inputs and the supplied PCE predictor."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from perovseek.bayesian import recommend
from perovseek.spectra import load_spectra, SpectralPredictor, prediction_metrics

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formulations = pd.read_excel(ROOT / "data/formulations.xlsx")
        cls.bounds = pd.read_excel(ROOT / "data/formulations.xlsx", sheet_name="Bounds")

    def test_recommendations_respect_excel_bounds_and_solvent_total(self):
        candidates = recommend(self.formulations, bounds=self.bounds)
        self.assertEqual(len(candidates), 6)
        components = self.formulations.columns.drop("PCE")
        limits = self.bounds.set_index("Component").loc[components]
        values = candidates[components].to_numpy()
        self.assertTrue((values >= limits.Lower.to_numpy() - 1e-6).all())
        self.assertTrue((values <= limits.Upper.to_numpy() + 1e-6).all())
        np.testing.assert_allclose(candidates[["DMF", "NFM", "EA"]].sum(axis=1), 1, atol=1e-6)
        self.assertNotIn("PCE", candidates.columns)

    def test_missing_measured_pce_is_rejected(self):
        incomplete = self.formulations.copy()
        incomplete.loc[0, "PCE"] = np.nan
        with self.assertRaisesRegex(ValueError, "complete"):
            recommend(incomplete, bounds=self.bounds)

    def test_infeasible_solvent_bounds_are_rejected(self):
        invalid = self.bounds.copy()
        invalid.loc[invalid.Component.isin(["DMF", "NFM", "EA"]), "Upper"] = 0.2
        invalid.loc[invalid.Component.isin(["DMF", "NFM", "EA"]), "Lower"] = 0
        with self.assertRaisesRegex(ValueError, "total fraction"):
            recommend(self.formulations, bounds=invalid)

    def test_supplied_predictor_matches_reference_test_results(self):
        spectra = load_spectra(ROOT / "data/spectra/1.59eV_additive_data.xlsx")
        model = SpectralPredictor(ROOT / "checkpoints/spectral_pce_state.pt")
        predictions = model.predict(spectra, subset="test")
        metrics = prediction_metrics(predictions)
        self.assertEqual(len(predictions), 1500)
        self.assertEqual(predictions.iloc[0].sample_id, "6-5-1")
        self.assertAlmostEqual(metrics["mae_pp"], 2.26576016, places=5)
        self.assertEqual(metrics["negative_predictions"], 6)

    def test_unlabelled_spectra_preserve_sample_identifiers(self):
        columns = ["wavelength", "film-A", "film-B"]
        absorption = pd.DataFrame({columns[0]: np.arange(400, 1002, 2),
                                   columns[1]: 1.0, columns[2]: 2.0})
        pl = pd.DataFrame({columns[0]: np.arange(700, 832, 2),
                          columns[1]: 10.0, columns[2]: 20.0})
        tables = {"Abs": absorption, "PL_top": pl, "PL_bottom": pl.copy()}
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.bin"
            source.write_bytes(b"test source fingerprint")
            with patch("perovseek.spectra.pd.read_excel", return_value=tables):
                spectra = load_spectra(source)
        self.assertEqual(spectra.sample_ids.tolist(), ["film-A", "film-B"])
        self.assertIsNone(spectra.measured_pce)


if __name__ == "__main__":
    unittest.main()
