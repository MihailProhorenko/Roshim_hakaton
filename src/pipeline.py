"""Chronological model selection, horizon diagnostics and forecast revision study."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pvc-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/pvc-cache')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='pvc-horizon-v1'
import matplotlib.pyplot as plt

from baseline import ROOT, H, load, smape, risk, strategy, features
from models import NAMES, forecast, error_scale, design


def block_origins(y, count=24):
    result, endpoint = [], len(y)
    for _ in range(count):
        origin = endpoint-H
        while origin>0 and pd.isna(y.iloc[origin-1]):
            origin -= 1
        if origin < 156:
            break
        result.append(origin)
        endpoint = origin
    return sorted(result)


def select_model(completed, cutoff=None):
    """Fixed rule: mean sMAPE on the last eight completed 13-week blocks."""
    if not completed:
        return 'naive'
    table = pd.DataFrame(completed)
    if cutoff is not None:
        table = table[table.origin+H <= cutoff]
    if table.empty:
        return 'naive'
    origins = sorted(table.origin.unique())[-8:]
    return str(table[table.origin.isin(origins)].groupby('model').smape.mean().sort_values().index[0])


def conformal_width(errors, scale):
    # Finite-sample rank; fewer than nine calibration scores cannot support 90%.
    values = np.sort(np.asarray(errors, dtype=float))
    rank = math.ceil((len(values)+1)*.9)
    return float(values[rank-1]*scale) if len(values)>=9 and rank<=len(values) else float('nan')


def main(path, refresh=True):
    out = ROOT/'artifacts'
    out.mkdir(exist_ok=True)
    frame, catalog, y, wide = load(path)
    catalog.to_csv(out/'series_catalog.csv')
    y.rename('k').to_csv(out/'target_history.csv')
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    code_hash = hashlib.sha256(b''.join((ROOT/'src'/name).read_bytes()
                                      for name in ['baseline.py','models.py','pipeline.py'])).hexdigest()[:12]
    cache_path = out/'cache'/f'{source_hash[:12]}-{code_hash}.json'
    cache_path.parent.mkdir(exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def predict(name, origin):
        key = f'{name}:{origin}'
        if key not in cache:
            result = forecast(name, y.iloc[:origin], wide.iloc[:origin])
            if result.shape != (H,) or not np.isfinite(result).all() or (result<=0).any():
                raise ValueError(f'Invalid forecast {key}')
            cache[key] = result.tolist()
            cache_path.write_text(json.dumps(cache))
        return np.array(cache[key])

    origins = block_origins(y)
    scores, details, policy, calibration = [], [], [], []
    strategy_records = []
    thresholds = [0., .0025, .005, .01]
    for number, origin in enumerate(origins):
        actual = y.iloc[origin:origin+H].to_numpy()
        previous = float(y.iloc[origin-1])
        scale = error_scale(y.iloc[:origin])
        # Choices are locked BEFORE observing this block.
        selected = select_model(scores, origin)
        previous_strategy = pd.DataFrame(strategy_records)
        threshold = .005
        if len(previous_strategy):
            recent = previous_strategy[previous_strategy.origin.isin(sorted(previous_strategy.origin.unique())[-8:])]
            ranking = recent.groupby('threshold').utility.mean()
            threshold = float(ranking.sort_values(ascending=False).index[0])
        for name in NAMES:
            prediction = predict(name, origin)
            scores.append(dict(origin=origin, origin_date=str(y.index[origin-1].date()),
                               model=name, smape=smape(actual,prediction),
                               mae=float(np.nanmean(np.abs(actual-prediction)))))
            for h in range(H):
                details.append(dict(origin=origin, origin_date=str(y.index[origin-1].date()),
                                    target_date=str(y.index[origin+h].date()), model=name,
                                    horizon=h+1, actual=actual[h], forecast=prediction[h],
                                    previous=previous, scale=scale))
        prediction = predict(selected, origin)
        for h in range(H):
            available = [r['score'] for r in calibration if r['model']==selected and r['horizon']==h+1]
            width = conformal_width(available[-20:], scale)
            policy.append(dict(origin=origin, origin_date=str(y.index[origin-1].date()),
                               target_date=str(y.index[origin+h].date()), model=selected,
                               horizon=h+1, actual=actual[h], forecast=prediction[h],
                               lower=max(0.,prediction[h]-width) if np.isfinite(width) else np.nan,
                               upper=prediction[h]+width, threshold=threshold,
                               evaluation=number>=4))
        # Update calibration and strategy only AFTER scoring this completed block.
        for name in NAMES:
            prediction_for_name = predict(name, origin)
            for h in range(H):
                if np.isfinite(actual[h]):
                    calibration.append(dict(model=name, horizon=h+1,
                                            score=abs(actual[h]-prediction_for_name[h])/scale))
        deltas = np.diff(np.r_[previous,actual])
        valid = np.isfinite(deltas)
        for candidate_threshold in thresholds:
            pnl = (strategy(prediction,previous,candidate_threshold)*deltas)[valid]
            drawdown = risk(pnl)
            strategy_records.append(dict(origin=origin, threshold=candidate_threshold,
                                         pnl=float(pnl.sum()), drawdown=drawdown,
                                         utility=float(pnl.sum()-drawdown)))
        print(f'Block {number+1}/{len(origins)} {y.index[origin-1].date()}: selected {selected}',flush=True)

    detail = pd.DataFrame(details)
    score = pd.DataFrame(scores)
    prequential = pd.DataFrame(policy)
    detail.to_csv(out/'validation_predictions.csv',index=False)
    score.to_csv(out/'validation_folds.csv',index=False)
    score.groupby('model').smape.agg(['mean','median','std','max']).sort_values('mean').to_csv(out/'leaderboard.csv')
    # Last eight completed blocks determine FINAL model, not the historical choices.
    name = select_model(scores, len(y))
    last_origins = origins[-8:]
    score[score.origin.isin(last_origins)].groupby('model').smape.mean().sort_values().to_csv(out/'model_selection.csv')
    prequential.to_csv(out/'prequential_predictions.csv',index=False)
    evaluated = prequential[prequential.evaluation & prequential.actual.notna()].copy()
    rows = []
    matched = detail[detail.actual.notna() & detail.origin.isin(origins[4:])]
    for model, group in matched.groupby('model'):
        for horizon, g in group.groupby('horizon'):
            rows.append(dict(model=model,horizon=horizon,n=len(g),
                             smape=smape(g.actual.to_numpy(),g.forecast.to_numpy()),
                             mae=float(np.mean(np.abs(g.actual-g.forecast)))))
    for horizon,g in evaluated.groupby('horizon'):
        rows.append(dict(model='adaptive_policy',horizon=horizon,n=len(g),
                         smape=smape(g.actual.to_numpy(),g.forecast.to_numpy()),
                         mae=float(np.mean(np.abs(g.actual-g.forecast)))))
    horizons = pd.DataFrame(rows)
    horizons.to_csv(out/'horizon_metrics.csv',index=False)
    complete_origins = [o for o in origins[4:] if y.iloc[o:o+H].notna().all()]
    balanced_rows=[]
    for model,group in detail[detail.origin.isin(complete_origins)].groupby('model'):
        for horizon,g in group.groupby('horizon'):
            balanced_rows.append(dict(model=model,horizon=horizon,n=len(g),
                                      smape=smape(g.actual.to_numpy(),g.forecast.to_numpy()),
                                      mae=float(np.mean(np.abs(g.actual-g.forecast)))))
    for horizon,g in prequential[prequential.origin.isin(complete_origins)].groupby('horizon'):
        balanced_rows.append(dict(model='adaptive_policy',horizon=horizon,n=len(g),
                                  smape=smape(g.actual.to_numpy(),g.forecast.to_numpy()),
                                  mae=float(np.mean(np.abs(g.actual-g.forecast)))))
    balanced=pd.DataFrame(balanced_rows)
    balanced.to_csv(out/'balanced_horizon_metrics.csv',index=False)
    ablations = []
    for base, external in [('ridge','ridge_external'),('catboost','catboost_external'),
                           ('catboost_mae','catboost_mae_external')]:
        paired = score[score.model.eq(base)].merge(score[score.model.eq(external)],on='origin',suffixes=('_base','_external'))
        paired = paired[paired.origin.isin(origins[4:])]
        ablations.append(dict(base=base,external=external,blocks=len(paired),
                              base_smape=float(paired.smape_base.mean()),
                              external_smape=float(paired.smape_external.mean()),
                              improvement_pp=float((paired.smape_base-paired.smape_external).mean()),
                              improved_blocks=int((paired.smape_external<paired.smape_base).sum())))
    pd.DataFrame(ablations).to_csv(out/'ablation_metrics.csv',index=False)
    comparisons = []
    for origin,g in evaluated.groupby('origin'):
        naive = detail[(detail.origin==origin)&(detail.model=='naive')&detail.actual.notna()]
        comparisons.append(dict(origin_date=g.origin_date.iloc[0],model=g.model.iloc[0],
                                adaptive_smape=smape(g.actual.to_numpy(),g.forecast.to_numpy()),
                                naive_smape=smape(naive.actual.to_numpy(),naive.forecast.to_numpy())))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out/'prequential_scores.csv',index=False)

    # Paired same-target revision: t+h forecast at t versus t+1 (h-1 ahead).
    revisions = []
    if refresh:
        for number, origin in enumerate(origins[4:]):
            selected = str(prequential.loc[prequential.origin.eq(origin),'model'].iloc[0])
            fixed = predict(selected,origin)
            if pd.notna(y.iloc[origin]):
                revised = predict(selected,origin+1)
                for h in range(1,H):
                    truth = y.iloc[origin+h]
                    if pd.notna(truth):
                        revisions.append(dict(origin=origin,model=selected, horizon=h+1,
                                              updated_origin=origin+1,updated_horizon=h,
                                              target_date=str(y.index[origin+h].date()),
                                              actual=truth, fixed=fixed[h], updated=revised[h-1],
                                              experiment='one_update'))
            # Weekly receding forecast of the SAME targets and SAME model family.
            for h in range(H):
                if pd.notna(y.iloc[origin+h-1]) and pd.notna(y.iloc[origin+h]):
                    revised = predict(selected,origin+h)[0]
                    revisions.append(dict(origin=origin,model=selected,horizon=h+1,
                                          updated_origin=origin+h,updated_horizon=1,
                                          target_date=str(y.index[origin+h].date()),
                                          actual=y.iloc[origin+h],fixed=fixed[h],updated=revised,
                                          experiment='weekly_updates'))
            print(f'Revision block {number+1}/{len(origins)-4}',flush=True)
    revision = pd.DataFrame(revisions)
    revision.to_csv(out/'revision_predictions.csv',index=False)
    revision_summary = []
    if len(revision):
        for (experiment,horizon),g in revision.groupby(['experiment','horizon']):
            fixed_error = smape(g.actual.to_numpy(),g.fixed.to_numpy())
            updated_error = smape(g.actual.to_numpy(),g.updated.to_numpy())
            revision_summary.append(dict(experiment=experiment,horizon=horizon,n=len(g),
                                         fixed_smape=fixed_error,updated_smape=updated_error,
                                         improvement_pp=fixed_error-updated_error,
                                         improved_fraction=float((np.abs(g.actual-g.updated)<np.abs(g.actual-g.fixed)).mean())))
    revision_table = pd.DataFrame(revision_summary)
    revision_table.to_csv(out/'revision_metrics.csv',index=False)

    # Evaluate selected strategy with thresholds selected only on past completed blocks.
    pnl_rows = []
    for origin,g in prequential[prequential.evaluation].groupby('origin'):
        previous=float(y.iloc[origin-1])
        delta=np.diff(np.r_[previous,g.actual.to_numpy()])
        positions=strategy(g.forecast.to_numpy(),previous,float(g.threshold.iloc[0]))
        for h in range(H):
            if np.isfinite(delta[h]):
                pnl_rows.append(dict(origin=origin,horizon=h+1,pnl=positions[h]*delta[h],
                                     buy_hold=delta[h],position=int(positions[h])))
    pnl_frame=pd.DataFrame(pnl_rows)
    pnl_frame.to_csv(out/'strategy_backtest.csv',index=False)
    strat=pd.DataFrame(strategy_records)
    strat.groupby('threshold')[['pnl','drawdown','utility']].sum().to_csv(out/'strategy_validation.csv')
    recent=strat[strat.origin.isin(last_origins)]
    threshold=float(recent.groupby('threshold').utility.mean().sort_values(ascending=False).index[0])

    final=predict(name,len(y))
    if name.startswith('catboost'):
        exported=forecast(name,y,wide,export_dir=out/'model')
        if not np.allclose(final,exported,rtol=1e-12,atol=1e-12):
            raise ValueError('Exported model does not reproduce final forecast')
    else:
        pd.DataFrame(columns=['feature','importance','label']).to_csv(out/'feature_importance.csv',index=False)
    scale=error_scale(y)
    widths=np.array([conformal_width([r['score'] for r in calibration if r['model']==name and r['horizon']==h][-20:],scale)
                     for h in range(1,H+1)])
    dates=pd.date_range(y.index[-1]+pd.Timedelta(days=7),periods=H,freq='7D')
    weeks=dates-pd.to_timedelta(dates.weekday,unit='D')
    submission=pd.DataFrame(dict(week_start=weeks.strftime('%Y-%m-%d'),horizon=np.arange(1,H+1),forecast=final))
    submission.to_csv(out/'submission.csv',index=False)
    positions=strategy(final,float(y.iloc[-1]),threshold)
    pd.DataFrame(dict(week_start=submission.week_start,position=positions)).to_csv(out/'strategy.csv',index=False)
    submission.assign(observation_date=dates.strftime('%Y-%m-%d'),lower=np.maximum(0.,final-widths),upper=final+widths,
                      position=positions).to_csv(out/'forecast_details.csv',index=False)
    coverage=evaluated[evaluated.lower.notna()].copy()
    covered=(coverage.actual>=coverage.lower)&(coverage.actual<=coverage.upper)
    interval_rows=[]
    for h,g in coverage.groupby('horizon'):
        interval_rows.append(dict(horizon=h,n=len(g),coverage=float(((g.actual>=g.lower)&(g.actual<=g.upper)).mean()),
                                  mean_width=float((g.upper-g.lower).mean())))
    pd.DataFrame(interval_rows).to_csv(out/'interval_metrics.csv',index=False)
    selected_features=design(y,wide,True)[1]
    # Actual external-quote identities are kept local with the catalog.
    pd.DataFrame({'feature':selected_features}).to_csv(out/'selected_features.csv',index=False)
    metadata=dict(target='ПВХ SG-5 (Китай) DAP Достык, USD/т',model=name,rows=len(frame),series=len(catalog),
                  target_observations=int(y.notna().sum()),target_calendar_weeks=len(y),
                  target_missing_weeks=int(y.isna().sum()),target_last_date=str(y.index[-1].date()),
                  source_last_date=str(frame.report_date.max().date()),input_sha256=source_hash,
                  code_hash=code_hash,validation_blocks=len(origins),evaluation_blocks=len(comparison),horizon=H,
                  selection_rule='Lowest mean sMAPE on last 8 fully completed blocks; first 4 blocks warm-up.',
                  calendar_note='Weekly Friday observations; week_start is the containing Monday.',
                  interval_coverage=float(covered.mean()) if len(coverage) else None,interval_n=len(coverage),
                  strategy_threshold=threshold,
                  strategy_pnl=float(pnl_frame.pnl.sum()),buy_hold_pnl=float(pnl_frame.buy_hold.sum()),
                  strategy_drawdown=risk(pnl_frame.pnl.to_numpy()),buy_hold_drawdown=risk(pnl_frame.buy_hold.to_numpy()),
                  limitations=['Historical chronological backtest, not an untouched competition test.',
                               'Weekly updates are a separate deployment experiment, not fixed-origin submission scoring.',
                               'Report dates assumed to be availability dates; values older than 45 days excluded.',
                               'Contract scenarios excluded due to ambiguous same-date observations.',
                               'Intervals are empirical, dependent and regime-sensitive; coverage is measured, not guaranteed.',
                               'Official CSV calendar pending template; model features end at last target observation.'])
    (out/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    plot_diagnostics(out,balanced,revision_table)
    write_report(out,metadata,comparison,balanced,revision_table)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


def plot_diagnostics(out,horizons,revisions):
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name in ['naive','adaptive_policy','catboost_mae_external','catboost_mae']:
        g=horizons[horizons.model.eq(name)]
        ax.plot(g.horizon,g.smape,marker='o',label=name)
    ax.set(xlabel='Forecast horizon (weeks)',ylabel='sMAPE (%)',xticks=range(1,14))
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(out/'error_by_horizon.svg',metadata={'Date':None});plt.close(fig)
    if len(revisions):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for experiment,g in revisions.groupby('experiment'):
            ax.plot(g.horizon,g.updated_smape,marker='o',label=experiment)
        g=revisions[revisions.experiment.eq('weekly_updates')]
        ax.plot(g.horizon,g.fixed_smape,marker='o',label='fixed origin, matched targets')
        ax.set(xlabel='Original forecast horizon (weeks)',ylabel='Paired sMAPE (%)',xticks=range(1,14))
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(out/'forecast_revision.svg',metadata={'Date':None});plt.close(fig)


def write_report(out,metadata,comparison,horizons,revisions):
    lines=['# Результаты последовательной исторической проверки','',
           f"Проверочных блоков после разогрева: {len(comparison)}. Каждый блок — 13 недель.",
           'Модель для каждого блока выбирается только по ранее завершённым блокам.',
           'Все кандидаты и правила фиксированы в коде. Это историческая проверка, не конкурсный скрытый тест.','',
           f"Средний sMAPE по блокам: адаптивный выбор {comparison.adaptive_smape.mean():.3f}%, naive {comparison.naive_smape.mean():.3f}%.",
           f"Финальная модель: `{metadata['model']}`.",'',
           '## Ошибка по горизонту','',
           'Для сравнения горизонтов используются одни и те же блоки со всеми 13 фактическими наблюдениями. Это исключает изменение состава выборки из-за пропусков.',
           '', '| Неделя | n | sMAPE, % | MAE, k |','|---|---|---|---|']
    for _,r in horizons[horizons.model.eq('adaptive_policy')].iterrows():
        lines.append(f'| {int(r.horizon)} | {int(r.n)} | {r.smape:.3f} | {r.mae:.4f} |')
    lines+=['','## Польза обновления после t+1','',
            'Сравниваются прогнозы одной и той же фактической даты: исходный t+h и пересчитанный после получения t+1.',
            'Семейство модели сохраняется; пересчёт получает только реально поступившие данные.','',
            '| Исходный горизонт | n | Без обновления, % | После одного обновления, % | Разница, п.п. |',
            '|---|---|---|---|---|']
    if len(revisions):
        for _,r in revisions[revisions.experiment.eq('one_update')].iterrows():
            lines.append(f'| {int(r.horizon)} | {int(r.n)} | {r.fixed_smape:.3f} | {r.updated_smape:.3f} | {r.improvement_pp:.3f} |')
    lines+=['','Положительная разница означает улучшение. Улучшение среднего не гарантирует улучшения каждой отдельной недели.',
            'Еженедельный пересчёт анализируется отдельно: его качество нельзя выдавать за качество фиксированного 13-недельного прогноза.','',
            '## Интервалы и стратегия','',
            f"Покрытие эмпирических интервалов: {metadata['interval_coverage']:.1%} на {metadata['interval_n']} наблюдениях.",
            'Интервалы используют только ошибки завершённых блоков и масштаб текущей волатильности.',
            f"Исторический P&L: {metadata['strategy_pnl']:.5f}; buy & hold: {metadata['buy_hold_pnl']:.5f} условных единиц.",
            f"Просадка: {metadata['strategy_drawdown']:.5f}; buy & hold: {metadata['buy_hold_drawdown']:.5f}.",
            'Порог стратегии выбирается по прошлым завершённым блокам, по критерию P&L минус максимальная просадка.',
            'P&L учитывает только переходы с наблюдаемыми ценами на обеих соседних неделях. Это не эффект в рублях и не процент доходности.','']
    (ROOT/'docs'/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/raw/predictors.xlsb')
    parser.add_argument('--skip-refresh',action='store_true',help='Skip the more expensive revision experiment')
    args=parser.parse_args()
    main(args.data,not args.skip_refresh)
