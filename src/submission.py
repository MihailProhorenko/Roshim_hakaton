"""Official, template-driven fixed-origin competition export (28 October)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FORECAST_COLUMNS=['week_start','horizon','forecast','forecast_lower','forecast_upper']
STRATEGY_COLUMNS=['week_start','position']


def templates(directory=None):
    directory=Path(directory) if directory is not None else ROOT/'templates'
    forecast=pd.read_csv(directory/'forecast_template.csv',sep=';',encoding='utf-8-sig')
    strategy=pd.read_csv(directory/'strategy_template.csv',sep=';',encoding='utf-8-sig')
    if list(forecast.columns)!=FORECAST_COLUMNS or list(strategy.columns)!=STRATEGY_COLUMNS:
        raise ValueError('Official template schema changed')
    if len(forecast)!=13 or forecast.horizon.tolist()!=list(range(1,14)):
        raise ValueError('Expected exactly 13 official horizons')
    if not forecast.week_start.equals(strategy.week_start):
        raise ValueError('Forecast and strategy template dates differ')
    dates=pd.to_datetime(forecast.week_start,format='%Y-%m-%d',errors='raise')
    if not dates.dt.weekday.eq(4).all() or not dates.diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError('Expected weekly Friday template dates')
    return forecast,strategy


def validate(forecast,strategy,directory=None):
    expected,expected_strategy=templates(directory)
    if list(forecast.columns)!=FORECAST_COLUMNS or list(strategy.columns)!=STRATEGY_COLUMNS:
        raise ValueError('Incorrect output columns/order')
    if len(forecast)!=13 or len(strategy)!=13:
        raise ValueError('Exactly 13 forecasts and positions required')
    if not forecast.week_start.reset_index(drop=True).equals(expected.week_start):
        raise ValueError('Forecast dates do not match the official template')
    if not strategy.week_start.reset_index(drop=True).equals(expected_strategy.week_start):
        raise ValueError('Strategy dates do not match the official template')
    if forecast.horizon.tolist()!=expected.horizon.tolist():
        raise ValueError('Invalid horizons')
    values=forecast[['forecast','forecast_lower','forecast_upper']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (forecast.forecast<=0).any() or (forecast.forecast_lower<0).any():
        raise ValueError('Expected finite positive forecasts and nonnegative bounds')
    if not ((forecast.forecast_lower<=forecast.forecast)&(forecast.forecast<=forecast.forecast_upper)).all():
        raise ValueError('Bounds must contain the point forecast')
    if not strategy.position.isin([-1,0,1]).all():
        raise ValueError('Positions must be -1, 0 or 1')


def export(details,out=None,directory=None):
    out=Path(out) if out is not None else ROOT/'artifacts'
    forecast,strategy=templates(directory)
    details=details.sort_values('horizon').reset_index(drop=True)
    if details.horizon.tolist()!=list(range(1,14)):
        raise ValueError('Missing or duplicated internal horizons')
    if details.observation_date.tolist()!=forecast.week_start.tolist():
        raise ValueError('Model target dates differ from official template; do not relabel predictions')
    forecast['forecast']=details.forecast.to_numpy()
    forecast['forecast_lower']=details.lower.to_numpy()
    forecast['forecast_upper']=details.upper.to_numpy()
    strategy['position']=details.position.to_numpy()
    validate(forecast,strategy,directory)
    out.mkdir(exist_ok=True,parents=True)
    for name,table in [('forecast.csv',forecast),('submission.csv',forecast),('strategy.csv',strategy)]:
        table.to_csv(out/name,index=False,sep=';',encoding='utf-8-sig',float_format='%.15g')
    # Internal diagnostics retain comma-separated format; dates follow official naming.
    details['week_start']=forecast.week_start.to_numpy()
    details.to_csv(out/'forecast_details.csv',index=False)
    return forecast,strategy


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Export existing predictions without retraining')
    parser.add_argument('--artifacts',type=Path,default=ROOT/'artifacts')
    args=parser.parse_args()
    export(pd.read_csv(args.artifacts/'forecast_details.csv',float_precision='round_trip'),out=args.artifacts)
    print('Official export validated: 13 Friday dates, semicolon CSV, UTF-8 BOM, point forecasts, bounds and fixed positions.')
