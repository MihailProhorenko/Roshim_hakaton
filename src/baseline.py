"""Point-in-time, fixed-origin 13-week PVC forecasting pipeline."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from pyxlsb import open_workbook, convert_date
from scipy.optimize import minimize

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
    target_id = target.series_id.iloc[0]
    wide = wide.drop(columns=[target_id], errors='ignore')
    # Every row uses only observations dated at or before the target observation.
    release_dates = wide.notna().apply(lambda col: pd.Series(
        np.where(col, wide.index.to_numpy(), np.datetime64('NaT')), index=wide.index))
    calendar = wide.index.union(y.index)
    release_dates = release_dates.reindex(calendar).sort_index().ffill().reindex(y.index)
    wide = wide.reindex(calendar).sort_index().ffill().reindex(y.index)
    for column in wide:
        age = (pd.Series(y.index, index=y.index)-release_dates[column]).dt.days
        wide[column] = wide[column].where(age.le(45))
    wide.attrs['products'] = catalog['product'].to_dict()
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






def strategy(prediction, previous, threshold):
    changes = np.diff(np.r_[previous, prediction])
    return np.where(np.abs(changes)/previous > threshold, np.sign(changes), 0).astype(int)


def risk(pnl):
    equity = np.r_[0., np.cumsum(pnl)]
    return float((np.maximum.accumulate(equity)-equity).max())
