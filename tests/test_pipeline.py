import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pipeline import features, strategy, risk, smape


class ForecastSafetyTests(unittest.TestCase):
    def test_features_do_not_change_when_future_changes(self):
        dates = pd.date_range('2020-01-03', periods=120, freq='W-FRI')
        y = pd.Series(np.exp(np.arange(120)*.001), index=dates)
        wide = pd.DataFrame({'raw': y*2})
        before, before_external = features(y, wide)
        y.iloc[100:] *= 100
        wide.iloc[100:] *= 50
        after, after_external = features(y, wide)
        pd.testing.assert_frame_equal(before.iloc[:100], after.iloc[:100])
        pd.testing.assert_frame_equal(before_external.iloc[:100], after_external.iloc[:100])

    def test_positions_are_fixed_from_forecast(self):
        np.testing.assert_array_equal(strategy(np.array([1.1, 1.0, 1.001]), 1., .005), [1,-1,0])

    def test_metrics_ignore_unobserved_target(self):
        self.assertAlmostEqual(smape(np.array([1.,np.nan]), np.array([2.,999.])), 200/3)
        self.assertAlmostEqual(risk([1.,-2.,1.]), 2.)


if __name__ == '__main__':
    unittest.main()
