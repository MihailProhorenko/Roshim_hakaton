import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from finalize import choose_model,choose_strategy,positions_for,score_positions


class FinalizationSafetyTests(unittest.TestCase):
    def test_better_average_does_not_override_recent_regression(self):
        config=json.loads((ROOT/'config/finalization.json').read_text())
        table=pd.DataFrame([
            dict(model='catboost_mae_external',mean=5.,median=4.,worst=16.,recent8=4.8),
            dict(model='conservative_25',mean=4.7,median=3.2,worst=13.,recent8=5.)])
        self.assertEqual(choose_model(table,config),'catboost_mae_external')
        table.loc[1,'recent8']=4.7
        self.assertEqual(choose_model(table,config),'conservative_25')

    def test_positive_but_concentrated_strategy_is_rejected(self):
        table=pd.DataFrame([
            dict(rule='flat',pnl=0.,stress_pnl=0.,profitable_blocks=0,largest_positive_fraction=1.,stress_utility=0.,trades=0),
            dict(rule='confidence',pnl=.1,stress_pnl=.08,profitable_blocks=3,largest_positive_fraction=.72,stress_utility=.07,trades=9)])
        self.assertEqual(choose_strategy(table),'flat')
        table.loc[1,'largest_positive_fraction']=.4
        self.assertEqual(choose_strategy(table),'confidence')

    def test_no_position_cannot_lose_even_under_costs(self):
        p=positions_for('flat',np.ones(13),1.,np.ones(13),np.ones(13),[[] for _ in range(13)])
        pnl,activity=score_positions(p,np.linspace(1.,.5,13),1.,.001)
        np.testing.assert_array_equal(pnl,np.zeros(13))
        self.assertEqual(activity,0)

    def test_costs_include_both_entry_and_exit(self):
        pnl,_=score_positions(np.ones(13,dtype=int),np.ones(13),1.,.001)
        self.assertAlmostEqual(pnl.sum(),-.002)

    def test_missing_prices_do_not_generate_pnl(self):
        truth=np.ones(13);truth[4]=np.nan;truth[5]=100.
        pnl,_=score_positions(np.ones(13,dtype=int),truth,1.)
        self.assertEqual(pnl[4],0.)
        self.assertEqual(pnl[5],0.)
        self.assertEqual(pnl[6],-99.)

    def test_capped_strategy_uses_strongest_forecast_signal_only(self):
        prediction=np.r_[1.02,1.05,np.repeat(1.05,11)]
        errors=[[.02]*8 for _ in range(13)]
        pos=positions_for('confidence_half_cap1',prediction,1.,prediction,prediction,errors,.005)
        self.assertEqual(np.count_nonzero(pos),1)
        self.assertEqual(pos[1],1)

    def test_capped_strategy_waits_for_calibration(self):
        prediction=np.linspace(1.,2.,13)
        pos=positions_for('confidence_half_cap1',prediction,1.,prediction,prediction,[[] for _ in range(13)],.005)
        self.assertTrue(np.all(pos==0))

    def test_positive_total_does_not_override_negative_early_period(self):
        row=dict(rule='confidence_half_cap1',pnl=.1,stress_pnl=.08,profitable_blocks=5,
                 largest_positive_fraction=.37,stress_utility=.05,trades=12,
                 stress_without_best_block=.02,stress_first12=-.001,stress_recent8=.081)
        self.assertEqual(choose_strategy(pd.DataFrame([row])),'flat')


if __name__=='__main__':
    unittest.main()
