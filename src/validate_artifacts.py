"""Independent output and temporal-alignment checks after a complete pipeline run."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from baseline import ROOT, load, H
from models import forecast


def main():
    out=ROOT/'artifacts'
    meta=json.loads((out/'run_metadata.json').read_text())
    path=ROOT/'data/raw/predictors.xlsb'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['input_sha256']
    _,_,y,wide=load(path)
    submission=pd.read_csv(out/'submission.csv')
    positions=pd.read_csv(out/'strategy.csv')
    assert len(submission)==len(positions)==H
    assert submission.horizon.tolist()==list(range(1,H+1))
    assert submission.week_start.equals(positions.week_start)
    assert set(positions.position).issubset({-1,0,1})
    assert np.isfinite(submission.forecast).all() and (submission.forecast>0).all()
    expected=pd.date_range(y.index[-1]+pd.Timedelta(days=3),periods=H,freq='7D')
    assert submission.week_start.tolist()==expected.strftime('%Y-%m-%d').tolist()
    reproduction=forecast(meta['model'],y,wide)
    np.testing.assert_allclose(reproduction,submission.forecast,rtol=1e-10)
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
