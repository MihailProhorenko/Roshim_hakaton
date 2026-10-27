"""Point-in-time, fixed-origin 13-week PVC forecasting pipeline."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from pyxlsb import open_workbook, convert_date
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
H = 13


def load(path):
    with open_workbook(str(path)) as book:
        with book.get_sheet(1) as sheet:
            rows = sheet.rows()
            columns = [c.v for c in next(rows)]
            frame = pd.DataFrame(([c.v for c in r] for r in rows), columns=columns)
    frame['report_date'] = frame.report_date.map(convert_date)
    frame['k'] = pd.to_numeric(frame.k, errors='raise')
    if not np.isfinite(frame.k).all() or (frame.k <= 0).any():
        raise ValueError('Expected finite positive anonymized prices')
    keys = list(frame.columns[1:11])
    frame['series_id'] = frame[keys].fillna('').astype(str).agg('|'.join, axis=1).map(
        lambda s: hashlib.sha256(s.encode()).hexdigest()[:12])
    mask = (frame.nomenclature_ru.eq('ПВХ SG-5 (Китай)') &
            frame.prod_version.eq('ПВХ SG-5 (Китай) DAP Достык, USD/т') &
            frame.frequency.eq('еженедельно'))
    target = frame.loc[mask].sort_values('report_date')
    assert target.series_id.nunique() == 1
    # A few holiday releases occur before Friday. Map to that week's Friday,
    # keeping the latest release if the week contains more than one.
    target['week'] = target.report_date.dt.to_period('W-SUN').dt.start_time + pd.Timedelta(days=4)
    y = target.groupby('week').k.last()
    y = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq='W-FRI'))
    y.index.name = 'report_date'
    catalog = frame.groupby('series_id').agg(
        product=('prod_version', 'first'), frequency=('frequency', 'first'),
        start=('report_date', 'min'), end=('report_date', 'max'), n=('k', 'size'))
    # Contract rows contain conflicting same-date quotes and unknown release timing.
    # Keep them in the audit/catalog, exclude them from forecasting features.
    usable = frame[frame.scenario.eq('факт')].copy()
    duplicate_series = usable.loc[usable.duplicated(['series_id','report_date'], keep=False), 'series_id'].unique()
    usable = usable[~usable.series_id.isin(duplicate_series)]
    wide = usable.pivot(index='report_date', columns='series_id', values='k').sort_index()
    # Every row uses only observations dated at or before the target observation.
    wide = wide.reindex(wide.index.union(y.index)).sort_index().ffill().reindex(y.index)
    return frame, catalog, y, wide


def smape(y, p):
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return float('nan')
    return float(200 * np.mean(np.abs(y[valid]-p[valid]) / np.maximum(np.abs(y[valid])+np.abs(p[valid]), 1e-12)))


def holt(y, damping):
    def run(params, forecast=False):
        alpha, beta = params
        level, trend = y[0], 0.0
        errors = []
        for value in y[1:]:
            estimate = level + damping*trend
            errors.append((value-estimate)**2)
            previous = level
            level = alpha*value + (1-alpha)*estimate
            trend = beta*(level-previous) + (1-beta)*damping*trend
        if forecast:
            return level + trend*np.cumsum(damping**np.arange(1, H+1))
        return np.mean(errors)
    fit = minimize(run, [0.5, 0.05], bounds=[(0.001, .999), (0.001, .999)], method='L-BFGS-B')
    return run(fit.x, True)


def features(y, wide):
    log = np.log(y)
    x = pd.DataFrame(index=y.index)
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        x[f'change_{lag}'] = log.diff(lag)
    for window in [4, 13, 26, 52]:
        x[f'mean_gap_{window}'] = log-log.rolling(window).mean()
        x[f'vol_{window}'] = log.diff().rolling(window).std()
    week = y.index.isocalendar().week.to_numpy(dtype=float)
    x['sin'] = np.sin(2*np.pi*week/52.18)
    x['cos'] = np.cos(2*np.pi*week/52.18)
    # Raw cross-series levels are deliberately excluded.
    extra = np.log(wide).diff(4).add_prefix('external_')
    return x, extra


def ridge_forecast(y, x, alpha):
    predictions = []
    for horizon in range(1, H+1):
        response = np.log(y.shift(-horizon)/y)
        valid = response.notna() & x.notna().all(axis=1)
        if valid.sum() < 70:
            return np.repeat(y.iloc[-1], H)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x.loc[valid], response.loc[valid])
        predictions.append(y.iloc[-1]*np.exp(model.predict(x.iloc[[-1]])[0]))
    return np.array(predictions)


def candidates(y, wide):
    filled = y.ffill()
    values = filled.to_numpy()
    x, extra = features(filled, wide)
    out = {
        'naive': np.repeat(values[-1], H),
        'seasonal_naive': values[-52:-52+H],
        'holt': holt(values, 1.0),
        'damped_holt': holt(values, .95),
    }
    for alpha in [100., 1000.]:
        out[f'ridge_{int(alpha)}'] = ridge_forecast(y, x, alpha)
    # Feature eligibility and ranking use only this fold's training history.
    eligible = extra.columns[extra.tail(104).notna().mean().ge(.95) & extra.iloc[-1].notna()]
    ranking = extra[eligible].corrwith(np.log(y).diff(4)).abs().sort_values(ascending=False)
    chosen = ranking.dropna().head(8).index
    ex = pd.concat([x, extra[chosen]], axis=1)
    out['ridge_external'] = ridge_forecast(y, ex, 1000.)
    out['ensemble'] = .5*out['damped_holt'] + .5*out['ridge_external']
    return {k: np.maximum(v, 1e-8) for k, v in out.items()}


def strategy(prediction, previous, threshold):
    changes = np.diff(np.r_[previous, prediction])
    return np.where(np.abs(changes)/previous > threshold, np.sign(changes), 0).astype(int)


def risk(pnl):
    equity = np.r_[0., np.cumsum(pnl)]
    return float((np.maximum.accumulate(equity)-equity).max())


def main(path):
    destination = ROOT/'artifacts'
    destination.mkdir(exist_ok=True)
    frame, catalog, y, wide = load(path)
    catalog.to_csv(destination/'series_catalog.csv')
    # Eight disjoint 13-week validation blocks, all within supplied training data.
    origins = []
    endpoint = len(y)
    for _ in range(8):
        origin = endpoint-H
        while pd.isna(y.iloc[origin-1]):
            origin -= 1
        origins.append(origin)
        endpoint = origin
    origins.sort()
    records, forecasts = [], []
    for origin in origins:
        train = y.iloc[:origin]
        actual = y.iloc[origin:origin+H].to_numpy()
        for name, prediction in candidates(train, wide.iloc[:origin]).items():
            records.append(dict(origin=str(train.index[-1].date()), model=name,
                                smape=smape(actual, prediction),
                                mae=float(np.nanmean(np.abs(actual-prediction))),
                                error_h1=smape(actual[:1], prediction[:1]),
                                error_h4=smape(actual[3:4], prediction[3:4]),
                                error_h13=smape(actual[12:13], prediction[12:13])))
            for h, (truth, pred) in enumerate(zip(actual, prediction), 1):
                forecasts.append(dict(origin=str(train.index[-1].date()), model=name,
                                      horizon=h, actual=truth, forecast=pred,
                                      previous=train.iloc[-1]))
        print('Validated origin', train.index[-1].date(), flush=True)
    scores = pd.DataFrame(records)
    scores.to_csv(destination/'validation_folds.csv', index=False)
    detail = pd.DataFrame(forecasts)
    detail.to_csv(destination/'validation_predictions.csv', index=False)
    summary = scores.groupby('model').smape.agg(['mean', 'median', 'std', 'max']).sort_values('mean')
    summary.to_csv(destination/'leaderboard.csv')
    # Reserve the last two blocks for an untouched outer evaluation.
    selection_origins = sorted(scores.origin.unique())[:-2]
    selection = scores[scores.origin.isin(selection_origins)]
    selection_table = selection.groupby('model').smape.mean().sort_values()
    selection_table.to_csv(destination/'model_selection.csv')
    name = selection_table.index[0]
    forecast = candidates(y, wide)[name]
    selected = detail[detail.model.eq(name)]
    # Empirical marginal intervals, estimated on validation residuals; not calibrated guarantees.
    residual = selected[selected.origin.isin(selection_origins)].assign(
        error=lambda d: np.abs(d.actual-d.forecast)/d.previous)
    widths = residual.groupby('horizon').error.quantile(.9).to_numpy()*y.iloc[-1]
    outer = selected[~selected.origin.isin(selection_origins)].copy()
    relative_width = residual.groupby('horizon').error.quantile(.9)
    outer['covered'] = np.abs(outer.actual-outer.forecast) <= outer.horizon.map(relative_width)*outer.previous
    outer.to_csv(destination/'outer_evaluation.csv', index=False)
    dates = pd.date_range(y.index[-1]+pd.Timedelta(days=7), periods=H, freq='7D')
    weeks = dates-pd.to_timedelta(dates.weekday, unit='D')
    submission = pd.DataFrame(dict(week_start=weeks.strftime('%Y-%m-%d'),
                                   horizon=np.arange(1,H+1), forecast=forecast))
    assert len(submission)==H and np.isfinite(forecast).all()
    submission.to_csv(destination/'submission.csv', index=False)
    strategy_scores = []
    for threshold in [0., .0025, .005, .01]:
        pnl, buy = [], []
        for _, fold in selected.groupby('origin', sort=True):
            delta = np.diff(np.r_[fold.previous.iloc[0], fold.actual.to_numpy()])
            valid = np.isfinite(delta)
            pnl.extend((strategy(fold.forecast.to_numpy(), fold.previous.iloc[0], threshold)*delta)[valid])
            buy.extend(delta[valid])
        strategy_scores.append(dict(threshold=threshold, pnl=float(np.sum(pnl)),
                                    max_drawdown=risk(pnl), buy_hold=float(np.sum(buy)),
                                    buy_hold_drawdown=risk(buy)))
    pd.DataFrame(strategy_scores).to_csv(destination/'strategy_validation.csv', index=False)
    # Predetermined conservative rule: neutral below a 0.5% predicted weekly move.
    position = strategy(forecast, y.iloc[-1], .005)
    pd.DataFrame(dict(week_start=submission.week_start, position=position)).to_csv(destination/'strategy.csv', index=False)
    submission.assign(observation_date=dates.strftime('%Y-%m-%d'),
                      lower=np.maximum(0,forecast-widths), upper=forecast+widths,
                      position=position).to_csv(destination/'forecast_details.csv', index=False)
    y.rename('k').to_csv(destination/'target_history.csv')
    metadata = dict(target='ПВХ SG-5 (Китай) DAP Достык, USD/т', model=name,
                    rows=len(frame), series=len(catalog), target_observations=int(y.notna().sum()),
                    target_calendar_weeks=len(y), target_missing_weeks=int(y.isna().sum()),
                    target_last_date=str(y.index[-1].date()),
                    source_last_date=str(frame.report_date.max().date()),
                    input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    validation_blocks=len(origins), horizon=H,
                    calendar_note='Weekly Friday observations; week_start is the containing Monday.',
                    limitations=['Official dates and scorer pending confirmation.',
                                 'Selection uses first six blocks; last two blocks are outer evaluation.',
                                 'Intervals estimated on six blocks have limited calibration evidence.',
                                 'Report dates are assumed to be availability dates.',
                                 'Source rows after last target Friday are excluded from model features.'])
    (destination/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(summary.to_string())
    print('Selected:', name)
    print(submission.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/raw/predictors.xlsb')
    args = parser.parse_args()
    main(args.data)
