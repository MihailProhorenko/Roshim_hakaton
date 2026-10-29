"""Bounded finalization: preserve incumbent unless all declared guards pass."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import ROOT,H,load,smape,risk,strategy
from models import forecast,error_scale
from pipeline import conformal_width
from submission import export,validate


def final_predict(name,y,wide):
    if name in ['conservative_25','conservative_50']:
        weight=.25 if name.endswith('25') else .5
        return (1-weight)*forecast('naive',y,wide)+weight*forecast('catboost_mae',y,wide)
    return forecast(name,y,wide)


def choose_model(table,config):
    incumbent=config['incumbent']
    baseline=table.set_index('model').loc[incumbent]
    accepted=[]
    for row in table.itertuples():
        if row.model not in config['candidates']:
            continue
        guards=config['acceptance']
        if (row.mean<=baseline['mean'] and row.worst<=baseline['worst'] and
            row.median<=baseline['median']+guards['median_smape_tolerance_pp'] and
            row.recent8<=baseline['recent8']+guards['recent8_smape_tolerance_pp']):
            accepted.append(row.model)
    if not accepted:
        return incumbent
    return table[table.model.isin(accepted)].sort_values(['worst','mean']).model.iloc[0]


def choose_strategy(table):
    # No active rule is accepted merely for losing less than buy-and-hold.
    valid=table[(table.rule!='flat')&(table.pnl>0)&(table.stress_pnl>0)&
                (table.profitable_blocks>=3)&(table.largest_positive_fraction<=.5)]
    return 'flat' if valid.empty else valid.sort_values(['stress_utility','trades'],ascending=[False,True]).rule.iloc[0]


def positions_for(rule,prediction,previous,first,second,errors,threshold=.005):
    change=np.diff(np.r_[previous,prediction])/previous
    if rule=='flat':
        return np.zeros(H,dtype=int)
    if rule=='threshold':
        return np.where(np.abs(change)>threshold,np.sign(change),0).astype(int)
    if rule=='agreement':
        a=np.sign(np.diff(np.r_[previous,first]))
        b=np.sign(np.diff(np.r_[previous,second]))
        return np.where((a==b)&(a==np.sign(change))&(np.abs(change)>threshold),a,0).astype(int)
    if rule=='confidence':
        uncertainty=np.array([max(threshold,float(np.median(e))) if len(e)>=4 else np.inf for e in errors])
        return np.where(np.abs(change)>uncertainty,np.sign(change),0).astype(int)
    raise ValueError(rule)


def score_positions(positions,actual,previous,cost=0.):
    delta=np.diff(np.r_[previous,actual])
    valid=np.isfinite(delta)
    gross=np.where(valid,positions*np.nan_to_num(delta),0.)
    # Costs apply even where price observations are missing, including terminal close.
    turnover=np.abs(np.diff(np.r_[0,positions,0]))
    net=gross-cost*previous*turnover[:-1]
    net[-1]-=cost*previous*turnover[-1]
    return net,int(np.count_nonzero(positions))


def main(rebuild=False):
    out=ROOT/'artifacts'
    if rebuild or not (out/'validation_predictions.csv').exists():
        subprocess.run([sys.executable,str(ROOT/'src/pipeline.py')],cwd=ROOT,check=True)
    config=json.loads((ROOT/'config/finalization.json').read_text())
    meta=json.loads((out/'run_metadata.json').read_text())
    frame,catalog,y,wide=load(ROOT/'data/raw/predictors.xlsb')
    detail=pd.read_csv(out/'validation_predictions.csv',float_precision='round_trip')
    all_origins=sorted(detail.origin.unique())
    evaluation=all_origins[4:]
    matrices={name:g.pivot(index='origin',columns='horizon',values='forecast').sort_index()
              for name,g in detail.groupby('model')}
    for name,weight in [('conservative_25',.25),('conservative_50',.5)]:
        matrices[name]=(1-weight)*matrices['naive']+weight*matrices['catboost_mae']
    actual=detail.drop_duplicates(['origin','horizon']).pivot(index='origin',columns='horizon',values='actual').sort_index()
    summary=[]
    fold_rows=[]
    for name in [config['incumbent'],'naive']+config['candidates']:
        errors=[]
        for origin in evaluation:
            score=smape(actual.loc[origin].to_numpy(),matrices[name].loc[origin].to_numpy())
            errors.append(score)
            fold_rows.append(dict(origin=int(origin),model=name,smape=score))
        summary.append(dict(model=name,mean=float(np.mean(errors)),median=float(np.median(errors)),
                            worst=float(np.max(errors)),recent8=float(np.mean(errors[-8:]))))
    comparison=pd.DataFrame(summary)
    name=choose_model(comparison,config)
    comparison['selected']=comparison.model.eq(name)
    comparison.to_csv(out/'final_model_comparison.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'final_model_folds.csv',index=False)
    rules=config['strategy_candidates']
    records={rule:[] for rule in rules}
    gains={rule:[] for rule in rules}
    stresses={rule:[] for rule in rules}
    activity={rule:0 for rule in rules}
    change_errors=[[] for _ in range(H)]
    calibration=[[] for _ in range(H)]
    interval_rows=[]
    calibration_rows=[]
    for origin in all_origins:
        origin=int(origin)
        previous=float(y.iloc[origin-1])
        truth=actual.loc[origin].to_numpy()
        prediction=matrices[name].loc[origin].to_numpy()
        scale=error_scale(y.iloc[:origin])
        widths=np.array([conformal_width(e[-20:],scale) for e in calibration])
        for h in range(H):
            if origin in evaluation and np.isfinite(widths[h]) and np.isfinite(truth[h]):
                interval_rows.append(dict(origin=origin,horizon=h+1,
                                           covered=bool(abs(truth[h]-prediction[h])<=widths[h]),width=2*widths[h]))
        if origin in evaluation:
            for rule in rules:
                positions=positions_for(rule,prediction,previous,
                                         matrices['catboost_mae'].loc[origin].to_numpy(),
                                         matrices['catboost_mae_external'].loc[origin].to_numpy(),
                                         change_errors,config['strategy_threshold'])
                pnl,trades=score_positions(positions,truth,previous)
                stress,_=score_positions(positions,truth,previous,cost=.001)
                records[rule].extend(pnl)
                gains[rule].append(float(pnl.sum()))
                stresses[rule].extend(stress)
                activity[rule]+=trades
        # Calibration update occurs only after the entire block is complete.
        truth_change=np.diff(np.r_[previous,truth])/previous
        forecast_change=np.diff(np.r_[previous,prediction])/previous
        for h in range(H):
            if np.isfinite(truth_change[h]):
                change_errors[h].append(abs(truth_change[h]-forecast_change[h]))
            if np.isfinite(truth[h]):
                calibration[h].append(abs(truth[h]-prediction[h])/scale)
    strategy_rows=[]
    for rule in rules:
        positive=[g for g in gains[rule] if g>0]
        strategy_rows.append(dict(rule=rule,pnl=float(np.sum(records[rule])),drawdown=risk(records[rule]),
                                  stress_pnl=float(np.sum(stresses[rule])),stress_utility=float(np.sum(stresses[rule])-risk(stresses[rule])),
                                  profitable_blocks=len(positive),losing_blocks=sum(g<0 for g in gains[rule]),
                                  largest_positive_fraction=max(positive)/sum(positive) if positive else 1.,trades=activity[rule]))
    strategies=pd.DataFrame(strategy_rows)
    chosen=choose_strategy(strategies)
    strategies['selected']=strategies.rule.eq(chosen)
    strategies.to_csv(out/'final_strategy_comparison.csv',index=False)
    # No test facts enter final fit. Frozen candidate is fit to the full official history.
    prediction=final_predict(name,y,wide)
    if name.startswith('catboost'):
        reproduced=forecast(name,y,wide,export_dir=out/'model')
        np.testing.assert_allclose(prediction,reproduced,rtol=1e-12)
    first=forecast('catboost_mae',y,wide)
    second=forecast('catboost_mae_external',y,wide)
    positions=positions_for(chosen,prediction,float(y.iloc[-1]),first,second,change_errors,config['strategy_threshold'])
    scale=error_scale(y)
    widths=np.array([conformal_width(e[-20:],scale) for e in calibration])
    dates=pd.date_range(y.index[-1]+pd.Timedelta(days=7),periods=H,freq='7D').strftime('%Y-%m-%d')
    details=pd.DataFrame(dict(week_start=dates,observation_date=dates,horizon=range(1,H+1),forecast=prediction,
                              lower=np.maximum(0.,prediction-widths),upper=prediction+widths,position=positions))
    export(details,out=out)
    intervals=pd.DataFrame(interval_rows)
    intervals.groupby('horizon').agg(n=('covered','size'),coverage=('covered','mean'),mean_width=('width','mean')).to_csv(out/'interval_metrics.csv')
    previous_pnl=meta.get('legacy_strategy_pnl',meta['strategy_pnl'])
    research_coverage=meta.get('research_interval_coverage',meta['interval_coverage'])
    chosen_strategy=strategies.set_index('rule').loc[chosen]
    naive=comparison.set_index('model').loc['naive']
    selected=comparison.set_index('model').loc[name]
    meta.update(model=name,finalized=True,finalization_version='2026-10-29-v1',
                finalization_config_sha256=hashlib.sha256((ROOT/'config/finalization.json').read_bytes()).hexdigest(),
                finalization_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                final_strategy=chosen,strategy_threshold=config['strategy_threshold'] if chosen!='flat' else None,
                strategy_pnl=float(chosen_strategy.pnl),strategy_drawdown=float(chosen_strategy.drawdown),
                strategy_stress_pnl=float(chosen_strategy.stress_pnl),legacy_strategy_pnl=previous_pnl,
                research_interval_coverage=research_coverage,interval_coverage=float(intervals.covered.mean()),interval_n=len(intervals),
                final_model_mean_smape=float(selected['mean']),final_naive_mean_smape=float(naive['mean']),
                final_model_recent8_smape=float(selected.recent8),final_naive_recent8_smape=float(naive.recent8),
                final_worst_smape=float(selected.worst),
                model_decision='Incumbent retained because neither conservative blend passed all no-regression guards.' if name==config['incumbent'] else 'Conservative candidate passed all declared development guards.',
                strategy_decision='No active rule passed the positive-P&L, stress and concentration guards; fixed zero positions.' if chosen=='flat' else 'Active rule passed declared historical development guards.',
                selection_rule='Frozen final candidate selected by bounded development comparison; no weekly competition reselection.',
                evaluation_note='Historical development comparison, not an independent test. Trading profits are not guaranteed.')
    (out/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    write_final_report(meta,comparison,strategies)
    print(comparison.to_string(index=False))
    print(strategies.to_string(index=False))
    print('FROZEN:',name,chosen,flush=True)


def write_final_report(meta,comparison,strategies):
    lines=['# Финальная проверка перед сдачей — 29 октября','',
           f"Модель: `{meta['model']}`. Стратегия: `{meta['final_strategy']}`.",
           'Это ограниченное сравнение на уже исследованной истории; независимый конкурсный результат неизвестен.','',
           '## Кандидаты прогноза','',
           '| Кандидат | Средний sMAPE, % | Медиана | Худший блок | Последние 8 |',
           '|---|---|---|---|---|']
    for r in comparison.itertuples():
        lines.append(f'| {r.model} | {r.mean:.3f} | {r.median:.3f} | {r.worst:.3f} | {r.recent8:.3f} |')
    lines+=['','Правило замены: не ухудшить среднюю, медианную, худшую и недавнюю ошибку относительно резервной модели.',
            'Осторожные ансамбли улучшают часть показателей, но ухудшают недавние окна. Поэтому оставлен исходный прогноз.','',
            '## Стратегии','', '| Правило | P&L | Просадка | P&L при условных затратах 0,1% на оборот | Активных недель |',
            '|---|---|---|---|---|']
    for r in strategies.itertuples():
        lines.append(f'| {r.rule} | {r.pnl:.5f} | {r.drawdown:.5f} | {r.stress_pnl:.5f} | {r.trades} |')
    lines+=['',meta['strategy_decision'],
            'Нулевая позиция означает отсутствие сделок и нулевой P&L, а не прибыльную торговую систему.',
            'Затраты 0,1% — исследовательская чувствительность, не правило официального скорера.',
            f"Для замороженной модели покрытие интервалов: {meta['interval_coverage']:.1%} на {meta['interval_n']} наблюдениях.",
            'Уровень номинально 90%; ширина и зависимость ошибок ограничивают практическую интерпретацию.',
            'Отчёт отделяет результат замороженной модели от исследования автоматического переключения моделей.','',
            '## Сдача','', '29 октября, до 21:00 по Москве. forecast.csv и strategy.csv: ;, UTF-8 BOM, официальные 13 пятниц.',
            'Прогноз и позиции фиксированы один раз; пересчёт после новых фактов — отдельный исследовательский сценарий.','']
    (ROOT/'docs/FINALIZATION.md').write_text('\n'.join(lines))
    research=ROOT/'docs/RESULTS.md'
    if research.exists() and not research.read_text().startswith('> Исследовательский'):
        research.write_text('> Исследовательский отчёт до финального выбора. Итоговая конфигурация: [FINALIZATION.md](FINALIZATION.md).\n\n'+research.read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rebuild',action='store_true',help='Rebuild all historical research artifacts before final freeze')
    args=parser.parse_args()
    main(args.rebuild)
