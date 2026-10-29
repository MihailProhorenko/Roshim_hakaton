import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from preflight import scalar_ledger,decision


class IndependentAuditTests(unittest.TestCase):
    def test_short_trade_gross_and_entry_exit_cost(self):
        positions=np.zeros(13,dtype=int);positions[0]=-1
        gross,net=scalar_ledger(positions,np.repeat(.9,13),1.,.001)
        self.assertAlmostEqual(gross.sum(),.1)
        self.assertAlmostEqual(net.sum(),.098)

    def test_signal_cap_and_missing_calibration(self):
        prediction=np.r_[1.02,1.06,np.repeat(1.06,11)]
        errors=[[.02]*8 for _ in range(13)]
        position=decision(prediction,1.,errors,.5,.005)
        self.assertEqual(np.count_nonzero(position),1)
        self.assertEqual(position[1],1)
        np.testing.assert_array_equal(decision(prediction,1.,[[] for _ in range(13)],.5,.005),np.zeros(13))
