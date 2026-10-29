"""Final diagnostics, independently scored. Never selects or retunes a strategy."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from baseline import ROOT,H,load
from models import forecast


def scalar_ledger(position,actual,previous,cost):
    gross=[];net=[]
    previous_position=0
    for h in range(H):
        past=previous if h==0 else actual[h-1]
        gain=float(position[h])*(float(actual[h])-float(past)) if np.isfinite(past) and np.isfinite(actual[h]) else 0.
        fee=cost*previous*abs(int(position[h])-previous_position)
        gross.append(gain);net.append(gain-fee)
        previous_position=int(position[h])
    net[-1]-=cost*previous*abs(previous_position)
    return np.array(gross),np.array(net)


def decision(prediction,previous,calibration,alpha,floor):
    signal=np.diff(np.r_[previous,prediction])/previous
    threshold=np.array([max(floor,alpha*float(np.median(e))) if len(e)>=4 else np.inf for e in calibration])
    eligible=[h for h in range(H) if abs(signal[h])>threshold[h]]
    result=np.zeros(H,dtype=int)
    if eligible:
        best=max(eligible,key=lambda h:abs(signal[h])/threshold[h])
        result[best]=1 if signal[best]>0 else -1
    return result


def main():
    out=ROOT/'artifacts'
    cfg=json.loads((ROOT/'config/preflight.json').read_text())
    meta=json.loads((out/'run_metadata.json').read_text())
    originals={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['forecast.csv','strategy.csv']}
    _,_,y,wide=load(ROOT/'data/raw/predictors.xlsb')
    data=pd.read_csv(out/'validation_predictions.csv',float_precision='round_trip')
    data=data[data.model.eq(meta['model'])]
    origins=sorted(data.origin.unique());evaluation=origins[4:]
    history=[[] for _ in range(H)]
    records=[]
    original_backtest=pd.read_csv(out/'final_strategy_backtest.csv',float_precision='round_trip')
    for origin in origins:
        group=data[data.origin.eq(origin)].sort_values('horizon')
        actual=group.actual.to_numpy();prediction=group.forecast.to_numpy();previous=float(group.previous.iloc[0])
        assert np.isclose(previous,y.iloc[origin-1])
        np.testing.assert_allclose(actual,y.iloc[origin:origin+H],equal_nan=True,rtol=1e-12)
        if origin in evaluation:
            for case in cfg['cases']:
                positions=decision(prediction,previous,history,case['alpha'],case['floor'])
                gross,stress=scalar_ledger(positions,actual,previous,cfg['stress_cost'])
                assert np.count_nonzero(positions)<=1
                if case['name']=='frozen':
                    saved=original_backtest[original_backtest.origin.eq(origin)].sort_values('horizon')
                    np.testing.assert_array_equal(positions,saved.position)
                    np.testing.assert_allclose(gross,saved.pnl,atol=1e-12)
                    np.testing.assert_allclose(stress,saved.stress_pnl,atol=1e-12)
                records.append(dict(case=case['name'],origin=int(origin),origin_date=str(y.index[origin-1].date()),
                                    pnl=float(gross.sum()),stress_pnl=float(stress.sum()),active=int(np.count_nonzero(positions))))
        actual_change=np.diff(np.r_[previous,actual])/previous
        predicted_change=np.diff(np.r_[previous,prediction])/previous
        for h in range(H):
            if np.isfinite(actual_change[h]):history[h].append(abs(actual_change[h]-predicted_change[h]))
    record=pd.DataFrame(records)
    summary=record.groupby('case',sort=False).agg(pnl=('pnl','sum'),stress_pnl=('stress_pnl','sum'),active_weeks=('active','sum'))
    summary.to_csv(out/'preflight_sensitivity.csv')
    frozen=record[record.case.eq('frozen')].copy()
    removal=[]
    for count in [0,1,2]:
        # Check both gross-best and stressed-best removal; rankings may differ.
        for rank in ['pnl','stress_pnl']:
            keep=frozen.drop(frozen.nlargest(count,rank).index) if count else frozen
            removal.append(dict(removed=count,ranked_by=rank,remaining_blocks=len(keep),
                                pnl=float(keep.pnl.sum()),stress_pnl=float(keep.stress_pnl.sum())))
    removal=pd.DataFrame(removal);removal.to_csv(out/'preflight_outliers.csv',index=False)
    early=frozen.iloc[:12];recent=frozen.iloc[-8:]
    for origin in [evaluation[0],evaluation[-1]]:
        predicted=forecast(meta['model'],y.iloc[:origin],wide.iloc[:origin])
        saved=data[data.origin.eq(origin)].sort_values('horizon').forecast.to_numpy()
        np.testing.assert_allclose(predicted,saved,rtol=1e-10)
    final_prediction=forecast(meta['model'],y,wide)
    actual_csv=pd.read_csv(out/'strategy.csv',sep=';',encoding='utf-8-sig')
    positions=decision(final_prediction,float(y.iloc[-1]),history,.5,.005)
    np.testing.assert_array_equal(positions,actual_csv.position)
    sensitivity_positions={case['name']:decision(final_prediction,float(y.iloc[-1]),history,case['alpha'],case['floor']).tolist() for case in cfg['cases']}
    assert originals=={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in originals}
    result=dict(audit_passed=True,independent_scalar_scoring=True,original_csv_unchanged=True,
                all_neighbor_stress_positive=bool(summary.stress_pnl.gt(0).all()),
                early_stress_pnl=float(early.stress_pnl.sum()),recent_stress_pnl=float(recent.stress_pnl.sum()),
                current_positions_by_case=sensitivity_positions,
                limitations='Diagnostics on researched development history; not an independent performance test.')
    (out/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# Последние три проверки — 29 октября','',
           'Проверки диагностические: параметры и файлы сдачи не изменяются.','',
           '## 1. Чувствительность к порогу','', '| Вариант | P&L | С условными затратами | Активные недели |','|---|---|---|---|']
    for case,r in summary.iterrows():lines.append(f'| {case} | {r.pnl:+.5f} | {r.stress_pnl:+.5f} | {int(r.active_weeks)} |')
    lines+=['','## 2. Исключение лучших периодов','', '| Исключено | Ранжирование | P&L | С затратами |','|---|---|---|---|']
    for r in removal.itertuples():lines.append(f'| {r.removed} | {r.ranked_by} | {r.pnl:+.5f} | {r.stress_pnl:+.5f} |')
    lines += ['',f"Ранние 12 блоков, с затратами: {early.stress_pnl.sum():+.5f}. Последние 8: {recent.stress_pnl.sum():+.5f}.",'',
              '## 3. Независимый аудит','',
              'Позиции пересчитаны отдельным кодом; P&L и затраты проверены скалярным циклом, без вызова основного скорера.',
              'Исторические факты сверены с датасетом. Модель заново обучена на префиксах начала и конца проверки; прогнозы совпали.',
              'CSV сдачи остались побайтно неизменными. Максимум одна активная неделя на блок соблюдён.','',
              'Вывод: положительный результат не является доказательством устойчивости на любых периодах. Зависимость от лучших блоков и небольшая выборка должны быть раскрыты.','']
    (ROOT/'docs/PREFLIGHT.md').write_text('\n'.join(lines))
    print(summary.to_string());print(removal.to_string(index=False));print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
