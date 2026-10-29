"""Independent output and temporal-alignment checks after a complete pipeline run."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from baseline import ROOT, load, H
from models import forecast
from submission import validate
from finalize import final_predict,positions_for


def main():
    out=ROOT/'artifacts'
    meta=json.loads((out/'run_metadata.json').read_text())
    path=ROOT/'data/raw/predictors.xlsb'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['input_sha256']
    _,_,y,wide=load(path)
    submission=pd.read_csv(out/'forecast.csv',sep=';',encoding='utf-8-sig')
    positions=pd.read_csv(out/'strategy.csv',sep=';',encoding='utf-8-sig')
    frozen=out/'freeze_manifest.json'
    if frozen.exists():
        freeze=json.loads(frozen.read_text())
        for name,digest in freeze['official_csv_sha256'].items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
        for name,digest in freeze['decision_source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    validate(submission,positions)
    assert (out/'forecast.csv').read_bytes().startswith(b'\xef\xbb\xbf')
    assert (out/'strategy.csv').read_bytes().startswith(b'\xef\xbb\xbf')
    assert (out/'submission.csv').read_bytes()==(out/'forecast.csv').read_bytes()
    assert len(submission)==len(positions)==H
    assert submission.horizon.tolist()==list(range(1,H+1))
    assert submission.week_start.equals(positions.week_start)
    assert set(positions.position).issubset({-1,0,1})
    assert np.isfinite(submission.forecast).all() and (submission.forecast>0).all()
    expected=pd.date_range(y.index[-1]+pd.Timedelta(days=7),periods=H,freq='7D')
    assert submission.week_start.tolist()==expected.strftime('%Y-%m-%d').tolist()
    reproduction=final_predict(meta['model'],y,wide)
    np.testing.assert_allclose(reproduction,submission.forecast,rtol=1e-10)
    if meta.get('finalized'):
        assert hashlib.sha256((ROOT/'config/finalization.json').read_bytes()).hexdigest()==meta['finalization_config_sha256']
        assert hashlib.sha256((ROOT/'src/finalize.py').read_bytes()).hexdigest()==meta['finalization_code_sha256']
        if meta['final_strategy']=='flat':
            assert positions.position.eq(0).all()
            assert meta['strategy_pnl']==0 and meta['strategy_drawdown']==0
        else:
            backtest=pd.read_csv(out/'final_strategy_backtest.csv')
            assert np.isclose(backtest.pnl.sum(),meta['strategy_pnl'],atol=1e-12)
            assert np.isclose(backtest.stress_pnl.sum(),meta['strategy_stress_pnl'],atol=1e-12)
            if 'cap1' in meta['final_strategy']:
                assert backtest.groupby('origin').position.apply(lambda p:np.count_nonzero(p)).le(1).all()
            errors=[[] for _ in range(H)]
            history=pd.read_csv(out/'validation_predictions.csv',float_precision='round_trip')
            history=history[history.model.eq(meta['model'])]
            for origin,g in history.groupby('origin',sort=True):
                g=g.sort_values('horizon')
                previous=float(g.previous.iloc[0])
                actual_change=np.diff(np.r_[previous,g.actual.to_numpy()])/previous
                predicted_change=np.diff(np.r_[previous,g.forecast.to_numpy()])/previous
                for horizon in range(H):
                    if np.isfinite(actual_change[horizon]):
                        errors[horizon].append(abs(actual_change[horizon]-predicted_change[horizon]))
            expected_positions=positions_for(meta['final_strategy'],reproduction,float(y.iloc[-1]),
                                             forecast('catboost_mae',y,wide),forecast('catboost_mae_external',y,wide),
                                             errors,float(meta['strategy_threshold']))
            np.testing.assert_array_equal(expected_positions,positions.position)
            if 'cap1' in meta['final_strategy']:
                assert np.count_nonzero(expected_positions)<=1
    balanced=pd.read_csv(out/'balanced_horizon_metrics.csv')
    assert balanced.groupby('model').n.nunique().eq(1).all()
    revisions=pd.read_csv(out/'revision_predictions.csv')
    assert len(revisions)>0
    for r in revisions.itertuples():
        # Same target index reached from either forecast origin.
        target=int(r.origin+r.horizon-1)
        assert target==int(r.updated_origin+r.updated_horizon-1)
        assert str(y.index[target].date())==r.target_date
        assert pd.notna(y.iloc[r.updated_origin-1])
        assert np.isclose(r.actual,y.iloc[target],rtol=1e-12)
        assert r.updated_origin<=target
    # Refit representative revised models using ONLY the historical prefix.
    for index in sorted({0,len(revisions)//2,len(revisions)-1}):
        row=revisions.iloc[index]
        origin=int(row.updated_origin)
        prediction=forecast(row.model,y.iloc[:origin],wide.iloc[:origin])
        np.testing.assert_allclose(prediction[int(row.updated_horizon)-1],row.updated,rtol=1e-10)
    print(f'Artifacts valid: {H} forecasts; {len(revisions)} paired revisions aligned to the same targets; '
          'balanced horizons; final model and sampled updated forecasts independently reproduced.')


if __name__=='__main__':
    main()
