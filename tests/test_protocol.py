"""Pruebas de invariantes de evaluación y uso de información futura."""

import unittest

import numpy as np
import pandas as pd
import torch

from src import backtest as B
from src import data as D
from src.dataset import WindowDataset
from src.models.baselines import seasonal_naive_fn
from src.models.rnn import LSTMAutoregressive, LSTMMultiOutput, Seq2Seq
from src.train import _forward


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def panel(self):
        ds = pd.date_range("2014-09-20", "2014-10-02 23:00", freq="h")
        return pd.DataFrame(
            {
                "unique_id": np.repeat(["a", "b"], len(ds)),
                "ds": np.tile(ds, 2),
                "y": np.arange(len(ds) * 2, dtype=float),
            }
        )

    def test_test_labels_cannot_change_first_origin_forecast(self):
        p = self.panel()
        fn = seasonal_naive_fn(24)
        a = B.backtest(
            fn,
            p,
            start=D.TEST_START,
            end=D.TEST_START + pd.Timedelta(hours=23),
            verbose=False,
        )
        p.loc[p.ds >= D.TEST_START, "y"] = -1e9
        b = B.backtest(
            fn,
            p,
            start=D.TEST_START,
            end=D.TEST_START + pd.Timedelta(hours=23),
            verbose=False,
        )
        pd.testing.assert_frame_equal(a, b)

    def test_duplicate_forecasts_are_rejected_even_with_correct_row_count(self):
        def bad(hist, origin, H):
            fc = seasonal_naive_fn(24)(hist, origin, H)
            fc.loc[1] = fc.loc[0]
            return fc

        with self.assertRaisesRegex(AssertionError, "duplicadas"):
            B.backtest(
                bad,
                self.panel(),
                start=D.TEST_START,
                end=D.TEST_START + pd.Timedelta(hours=23),
                verbose=False,
            )

    def test_seq2seq_evaluation_ignores_targets(self):
        torch.manual_seed(42)
        m = Seq2Seq(hidden=8, teacher_forcing=1.0).eval()
        x, xf = torch.randn(3, 48, 8), torch.randn(3, 24, 7)
        a, _ = _forward(m, (x, xf, torch.zeros(3, 24)), True)
        b, _ = _forward(m, (x, xf, torch.full((3, 24), 1e6)), True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_recurrent_models_support_backprop_and_batch_independence(self):
        for cls in (LSTMAutoregressive, LSTMMultiOutput, Seq2Seq):
            m = cls(hidden=8)
            x, xf = torch.randn(3, 48, 8), torch.randn(3, 24, 7)
            m(x, xf).square().mean().backward()
            self.assertTrue(
                all(
                    torch.isfinite(p.grad).all()
                    for p in m.parameters()
                    if p.grad is not None
                )
            )
            m.eval()
            torch.testing.assert_close(
                m(x, xf)[0], m(x[:1], xf[:1])[0], rtol=1e-4, atol=1e-6
            )

    def test_ar_uses_one_step_targets_during_training(self):
        m = LSTMAutoregressive(hidden=8)
        self.assertEqual(m(torch.randn(4, 48, 8), torch.randn(4, 1, 7)).shape, (4, 1))

    def test_scaler_statistics_do_not_depend_on_validation_or_test(self):
        ds = pd.date_range("2014-06-20", "2014-07-02 23:00", freq="h")
        panel = pd.DataFrame(
            {"unique_id": "a", "ds": ds, "y": np.arange(len(ds), dtype=float)}
        )
        expected = D.SeriesScaler().fit(panel).stats
        panel.loc[panel.ds > D.TRAIN_END, "y"] = -1e12
        actual = D.SeriesScaler().fit(panel).stats
        pd.testing.assert_frame_equal(expected, actual)

    def test_windows_keep_all_targets_inside_their_temporal_split(self):
        ds = pd.date_range("2014-06-20", "2014-07-02 23:00", freq="h")
        panel = pd.DataFrame(
            {"unique_id": "a", "ds": ds, "y": np.arange(len(ds), dtype=float)}
        )
        cal = D.calendar_features(ds)
        for col in D.CALENDAR_COLS:
            panel[col] = cal[col].to_numpy()
        train = WindowDataset(panel, L=24, H=24, split="train")
        val = WindowDataset(panel, L=24, H=24, split="val", origin_hour=0)
        self.assertLessEqual(train.times[train.origin_idx.max() + 23], D.TRAIN_END)
        self.assertGreaterEqual(val.times[val.origin_idx.min()], D.VAL_START)
        first = val.origin_idx.min()
        np.testing.assert_allclose(val[0][0][:, 0], panel.y.iloc[first - 24 : first])


if __name__ == "__main__":
    unittest.main()
