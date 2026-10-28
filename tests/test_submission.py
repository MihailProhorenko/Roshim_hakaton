import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from submission import validate,export


class OfficialSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.forecast=pd.read_csv(ROOT/'templates/examples/forecast_example.csv',sep=';',encoding='utf-8-sig')
        self.strategy=pd.read_csv(ROOT/'templates/examples/strategy_example.csv',sep=';',encoding='utf-8-sig')

    def test_official_examples_match_contract(self):
        validate(self.forecast,self.strategy)

    def test_rejects_wrong_dates_bounds_positions_and_schema(self):
        for problem in ['monday','bounds','nan','position','schema','horizon']:
            with self.subTest(problem=problem):
                f,s=self.forecast.copy(),self.strategy.copy()
                if problem=='monday':f.loc[0,'week_start']='2026-07-06'
                if problem=='bounds':f.loc[0,'forecast_lower']=99.
                if problem=='nan':f.loc[0,'forecast_upper']=np.nan
                if problem=='position':s.loc[0,'position']=2
                if problem=='schema':f=f.drop(columns=['forecast_upper'])
                if problem=='horizon':f.loc[0,'horizon']=13
                with self.assertRaises(ValueError):validate(f,s)

    def test_export_semicolons_bom_and_roundtrip(self):
        f=self.forecast
        details=pd.DataFrame(dict(observation_date=f.week_start,week_start=f.week_start,
                                  horizon=f.horizon,forecast=f.forecast,
                                  lower=f.forecast_lower,upper=f.forecast_upper,position=self.strategy.position))
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            export(details,out=out)
            blob=(out/'forecast.csv').read_bytes()
            self.assertTrue(blob.startswith(b'\xef\xbb\xbf'))
            self.assertEqual(blob,(out/'submission.csv').read_bytes())
            self.assertIn('week_start;horizon;forecast;forecast_lower;forecast_upper',blob.decode('utf-8-sig'))
            f2=pd.read_csv(out/'forecast.csv',sep=';',encoding='utf-8-sig')
            s2=pd.read_csv(out/'strategy.csv',sep=';',encoding='utf-8-sig')
            validate(f2,s2)
            np.testing.assert_allclose(f2.forecast,f.forecast,rtol=1e-12)
            before={p.name:p.read_bytes() for p in out.glob('*.csv')}
            subprocess.run([sys.executable,str(ROOT/'src/submission.py'),'--artifacts',str(out)],check=True,capture_output=True)
            self.assertEqual(before,{p.name:p.read_bytes() for p in out.glob('*.csv')})
            changed=details.copy();changed.loc[0,'observation_date']='2026-07-17'
            with self.assertRaises(ValueError):export(changed,out=out)


if __name__=='__main__':
    unittest.main()
