import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pipeline import features, strategy, risk, smape, select_model, conformal_width, block_origins
from models import forecast


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

    def test_selection_ignores_uncompleted_block(self):
        scores = [dict(origin=100, model='naive', smape=2.),
                  dict(origin=100, model='holt', smape=3.),
                  dict(origin=120, model='naive', smape=100.),
                  dict(origin=120, model='holt', smape=0.)]
        self.assertEqual(select_model(scores, cutoff=125), 'naive')
        self.assertEqual(select_model(scores, cutoff=133), 'holt')

    def test_interval_rank_requires_enough_calibration(self):
        self.assertTrue(np.isnan(conformal_width([1.]*8,2.)))
        self.assertEqual(conformal_width(list(range(1,10)),2.),18.)

    def test_blocks_are_disjoint_and_origin_is_observed(self):
        dates=pd.date_range('2015-01-02',periods=564,freq='W-FRI')
        y=pd.Series(1.,index=dates)
        y.iloc[537]=np.nan
        origins=block_origins(y)
        self.assertEqual(len(origins),24)
        self.assertTrue(all(b-a>=13 for a,b in zip(origins,origins[1:])))
        self.assertTrue(all(pd.notna(y.iloc[o-1]) for o in origins))
        self.assertLessEqual(origins[-1]+13,len(y))

    def test_ridge_handles_missing_training_targets(self):
        dates=pd.date_range('2015-01-02',periods=200,freq='W-FRI')
        y=pd.Series(np.exp(np.sin(np.arange(200)/20)*.05),index=dates)
        y.iloc[80:83]=np.nan
        wide=pd.DataFrame({'raw':y*2})
        for name in ['ridge','ridge_external']:
            result=forecast(name,y,wide)
            self.assertEqual(result.shape,(13,))
            self.assertTrue(np.isfinite(result).all() and (result>0).all())


if __name__ == '__main__':
    unittest.main()
