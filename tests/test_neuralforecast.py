"""Integración del early stopping temporal con NeuralForecast instalado."""

import unittest

import numpy as np
import pandas as pd
import torch
from neuralforecast.models import NHITS

from src.data import TEST_START
from src.models.nf import fit_nf


class NeuralForecastTests(unittest.TestCase):
    def test_daily_validation_restores_hourly_training_stride(self):
        torch.set_num_threads(2)
        ds = pd.date_range("2014-09-01", periods=240, freq="h")
        panel = pd.DataFrame(
            {
                "unique_id": np.repeat(["a", "b"], len(ds)),
                "ds": np.tile(ds, 2),
                "y": np.tile(np.sin(np.arange(len(ds)) / 24), 2),
            }
        )

        def make(**kwargs):
            return NHITS(
                h=24,
                input_size=48,
                mlp_units=[[8, 8]] * 3,
                n_blocks=[1, 1, 1],
                batch_size=2,
                windows_batch_size=4,
                inference_windows_batch_size=8,
                accelerator="cpu",
                devices=1,
                **kwargs,
            )

        nf, info = fit_nf(
            make,
            panel,
            val_size=48,
            max_steps=2,
            val_check_steps=1,
            patience=2,
            validation_step_size=24,
        )
        self.assertEqual(nf.models[0].step_size, 1)
        self.assertEqual(len(info["valid_traj"]), 2)
        self.assertEqual(info["validation_step_size"], 24)
        self.assertTrue(np.isfinite(info["valid_traj"].val_loss).all())

    def test_early_stopping_rejects_test_dates(self):
        with self.assertRaisesRegex(ValueError, "anteriores al test"):
            fit_nf(lambda **kwargs: None, pd.DataFrame(), end=TEST_START)


if __name__ == "__main__":
    unittest.main()
