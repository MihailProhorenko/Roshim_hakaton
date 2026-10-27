"""Fixed, small candidate set. Every fit receives a historical prefix only."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import H, features, holt

NAMES = ['naive', 'seasonal_naive', 'holt', 'damped_holt', 'holt_104',
         'damped_holt_52', 'momentum_4', 'momentum_13', 'ridge',
         'ridge_208', 'ridge_external', 'catboost', 'catboost_external',
         'catboost_mae', 'catboost_mae_external', 'blend']


def design(y, wide, external=False):
    x, extra = features(y.ffill(), wide)
    log = np.log(y.ffill())
    x['log_target'] = log
    changed = y.ffill().diff().abs().gt(1e-9)
    x['unchanged_weeks'] = changed.groupby(changed.cumsum()).cumcount().astype(float)
    x['observed'] = y.notna().astype(float)
    selected = []
    if external:
        eligible = extra.columns[extra.tail(104).notna().mean().ge(.85) & extra.iloc[-1].notna()]
        response = np.log(y.shift(-4)/y)
        rank = extra[eligible].corrwith(response).abs().dropna().sort_values(ascending=False)
        selected = list(rank.head(6).index)
        ex = extra[selected]
        x = pd.concat([x, ex, ex.isna().astype(float).add_suffix('_missing')], axis=1)
        products = wide.attrs.get('products', {})
        definitions = {'pvc': ['PVC','ПВХ'], 'ethylene':['Ethylene','Этилен'],
                       'naphtha':['Naphtha','Нафта'], 'brent':['Brent']}
        for group, terms in definitions.items():
            columns = [c for c in wide if any(t.lower() in products.get(c,'').lower() for t in terms)]
            if not columns:
                continue
            for lag in [1,4,13]:
                market = np.log(wide[columns]).diff(lag).median(axis=1)
                x[f'{group}_return_{lag}'] = market
                x[f'{group}_divergence_{lag}'] = market-log.diff(lag)
    # Missingness does not look forward. Zero is neutral log-return/volatility.
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.), selected


def trend(y, window):
    changes = np.diff(np.log(y.ffill().tail(window+1)))
    # Bound unstable local slopes; horizon damping prevents indefinite extrapolation.
    slope = np.mean(changes)
    return y.iloc[-1]*np.exp(slope*np.cumsum(.85**np.arange(H)))


def forecast(name, y, wide, export_dir=None):
    if pd.isna(y.iloc[-1]):
        raise ValueError('Forecast origin must have an observed target')
    last = float(y.iloc[-1])
    values = y.ffill().to_numpy()
    if name == 'naive':
        return np.repeat(last, H)
    if name == 'seasonal_naive':
        return values[-52:-52+H]
    if name in ['holt', 'damped_holt', 'holt_104', 'damped_holt_52']:
        window = 104 if name == 'holt_104' else 52 if name == 'damped_holt_52' else len(y)
        return np.maximum(holt(values[-window:], .95 if 'damped' in name else 1.), 1e-8)
    if name.startswith('momentum'):
        return trend(y, int(name.split('_')[-1]))
    if name == 'blend':
        return .5*forecast('naive',y,wide)+.5*forecast('catboost_external',y,wide)
    x, _ = design(y, wide, external='external' in name)
    if name.startswith('ridge'):
        result = []
        for horizon in range(1, H+1):
            response = np.log(y.shift(-horizon)/y)
            valid = response.notna() & (np.arange(len(y)) >= 52)
            indices = np.flatnonzero(valid)
            if name == 'ridge_208':
                indices = indices[-208:]
            model = make_pipeline(StandardScaler(), Ridge(alpha=1000.))
            model.fit(x.iloc[indices], response.iloc[indices])
            result.append(last*np.exp(model.predict(x.iloc[[-1]])[0]))
        return np.maximum(result, 1e-8)
    if name.startswith('catboost'):
        blocks, labels = [], []
        for horizon in range(1,H+1):
            response = np.log(y.shift(-horizon)/y)
            indices = np.flatnonzero(response.notna() & (np.arange(len(y)) >= 52))[-260:]
            block = x.iloc[indices].copy()
            block['horizon'] = horizon
            blocks.append(block)
            labels.extend(response.iloc[indices])
        model = CatBoostRegressor(iterations=160, depth=4, learning_rate=.04,
                                  l2_leaf_reg=10., loss_function='MAE' if 'mae' in name else 'RMSE',
                                  random_seed=42, thread_count=1, verbose=False,
                                  allow_writing_files=False)
        model.fit(pd.concat(blocks), np.array(labels))
        if export_dir is not None:
            export_dir.mkdir(exist_ok=True,parents=True)
            model.save_model(str(export_dir/'model.cbm'))
            importance=pd.DataFrame({'feature':model.feature_names_,
                                     'importance':model.feature_importances_})
            products=wide.attrs.get('products',{})
            importance['label']=importance.feature.map(lambda f: products.get(f.removeprefix('external_'),f))
            importance.sort_values('importance',ascending=False).to_csv(export_dir.parent/'feature_importance.csv',index=False)
        future = pd.concat([x.iloc[[-1]]]*H, ignore_index=True)
        future['horizon'] = np.arange(1,H+1)
        return np.maximum(last*np.exp(model.predict(future)), 1e-8)
    raise ValueError(name)


def error_scale(y):
    """Price-unit risk scale based on observed training returns, with a fixed floor."""
    volatility = np.log(y).diff().tail(26).std()
    return float(y.iloc[-1]*max(float(volatility) if np.isfinite(volatility) else 0., .005))
