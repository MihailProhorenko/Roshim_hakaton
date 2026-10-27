"""Interactive forecast, risk and matched forecast-revision diagnostics."""
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART=Path(__file__).resolve().parent/'artifacts'
st.set_page_config(page_title='ПВХ · прогноз и решения',layout='wide')
st.title('ПВХ · прогноз и решения')
st.caption('SG-5 (Китай), DAP Достык · анонимизированный индекс k · 13 недель')
required=['run_metadata.json','forecast_details.csv','target_history.csv','prequential_scores.csv',
          'horizon_metrics.csv','balanced_horizon_metrics.csv','revision_metrics.csv','interval_metrics.csv','ablation_metrics.csv']
if not all((ART/name).exists() for name in required):
    st.info('Для подготовки результатов выполните python src/pipeline.py в окружении проекта.')
    st.stop()
meta=json.loads((ART/'run_metadata.json').read_text())
future=pd.read_csv(ART/'forecast_details.csv',parse_dates=['observation_date'])
history=pd.read_csv(ART/'target_history.csv',parse_dates=['report_date'])
comparison=pd.read_csv(ART/'prequential_scores.csv')
tabs=st.tabs(['Прогноз','Качество и риск','Обновление прогноза','Драйверы и применение'])

with tabs[0]:
    last=history.k.dropna().iloc[-1]
    a,b,c=st.columns(3)
    a.metric('Последний факт',f'{last:.4f}',help=meta['target_last_date'])
    b.metric('Через 13 недель',f'{future.forecast.iloc[-1]:.4f}',f'{100*(future.forecast.iloc[-1]/last-1):+.1f}%')
    c.metric('Выбранная модель',meta['model'])
    fig=go.Figure()
    recent=history.tail(104)
    fig.add_trace(go.Scatter(x=recent.report_date,y=recent.k,name='Факт',line=dict(color='#334155')))
    fig.add_trace(go.Scatter(x=future.observation_date,y=future.upper,name='Верхняя граница',line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=future.observation_date,y=future.lower,name='Эмпирический интервал',line=dict(width=0),fill='tonexty',fillcolor='rgba(37,99,235,.14)'))
    fig.add_trace(go.Scatter(x=future.observation_date,y=future.forecast,name='Прогноз',line=dict(color='#2563eb',dash='dash')))
    fig.update_layout(yaxis_title='Индекс k',hovermode='x unified',height=420,margin=dict(t=20,b=20))
    st.plotly_chart(fig,width='stretch')
    st.caption('Границы отражают ошибки прошлых прогнозов и текущую волатильность. Покрытие в будущем не гарантировано.')
    if meta['model']=='naive':
        st.info('По принятому правилу выбора сложные модели не превзошли сохранение последней цены на последних восьми исторических окнах. Плоский прогноз отражает отсутствие подтверждённого направленного сигнала.')
    st.dataframe(future.rename(columns={'forecast':'Прогноз','position':'Позиция','lower':'Нижняя граница','upper':'Верхняя граница'}),hide_index=True,width='stretch')
    for name in ['submission.csv','strategy.csv']:
        st.download_button(f'Скачать {name}',(ART/name).read_bytes(),file_name=name,mime='text/csv')

with tabs[1]:
    st.subheader('Как меняется ошибка с горизонтом')
    st.write('Каждый исторический прогноз фиксируется на все 13 недель. Выбор модели использует только уже завершённые проверочные блоки.')
    a,b=st.columns(2)
    a.metric('Последовательный выбор · sMAPE',f'{comparison.adaptive_smape.mean():.2f}%')
    b.metric('Naive на тех же блоках · sMAPE',f'{comparison.naive_smape.mean():.2f}%')
    if comparison.adaptive_smape.mean()>=comparison.naive_smape.mean():
        st.warning('Переключение моделей на этой исторической проверке уступило naive. Улучшение точности не подтверждено.')
    balanced=st.toggle('Одинаковые блоки для всех горизонтов',value=True)
    table=pd.read_csv(ART/('balanced_horizon_metrics.csv' if balanced else 'horizon_metrics.csv'))
    available=list(dict.fromkeys(['adaptive_policy','naive',meta['model'],'catboost_mae_external']))
    selected=st.multiselect('Модели для сравнения',list(table.model.unique()),default=[m for m in available if m in set(table.model)])
    fig=go.Figure()
    for name in selected:
        g=table[table.model.eq(name)]
        fig.add_trace(go.Scatter(x=g.horizon,y=g.smape,name=name,mode='lines+markers'))
    fig.update_layout(xaxis=dict(title='Горизонт, недели',dtick=1),yaxis_title='sMAPE, %',height=370,margin=dict(t=20))
    st.plotly_chart(fig,width='stretch')
    st.dataframe(table[table.model.isin(selected)],hide_index=True,width='stretch')
    st.subheader('Интервалы и стратегия')
    a,b,c=st.columns(3)
    a.metric('Покрытие интервалов',f"{meta['interval_coverage']:.1%}",help=f"{meta['interval_n']} исторических наблюдений с доступной калибровкой")
    b.metric('P&L стратегии',f"{meta['strategy_pnl']:.4f}")
    c.metric('P&L buy & hold',f"{meta['buy_hold_pnl']:.4f}")
    st.write(f"Максимальная просадка стратегии: {meta['strategy_drawdown']:.4f}; buy & hold: {meta['buy_hold_drawdown']:.4f}.")
    st.caption('P&L — условные единицы индекса, без комиссий и объёмов. Пропущенные недельные переходы исключены. Это не денежная экономия предприятия.')
    st.dataframe(pd.read_csv(ART/'interval_metrics.csv'),hide_index=True)

with tabs[2]:
    st.subheader('Помогает ли новый факт уточнить прогноз?')
    revision=pd.read_csv(ART/'revision_metrics.csv')
    mode=st.radio('Режим пересчёта',['После одной недели','Каждую неделю'],horizontal=True)
    experiment='one_update' if mode=='После одной недели' else 'weekly_updates'
    g=revision[revision.experiment.eq(experiment)]
    if experiment=='one_update':
        st.write('Сравниваем исходный прогноз t+h с прогнозом той же даты после получения факта t+1. Семейство модели сохраняется, модель обучается заново на доступной истории.')
    else:
        st.write('Перед каждой прогнозируемой неделей поступает предыдущий факт и выполняется новый одношаговый прогноз. Это сценарий эксплуатации с более полной информацией.')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=g.horizon,y=g.fixed_smape,name='Без пересчёта',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=g.horizon,y=g.updated_smape,name='С пересчётом',mode='lines+markers'))
    fig.update_layout(xaxis=dict(title='Исходный горизонт, недели',dtick=1),yaxis_title='sMAPE на совпадающих датах, %',height=370,margin=dict(t=20))
    st.plotly_chart(fig,width='stretch')
    st.dataframe(g,hide_index=True,width='stretch')
    st.caption('n — число пар прогнозов одной и той же даты. Положительная improvement_pp означает снижение средней ошибки. improved_fraction — доля случаев с меньшей абсолютной ошибкой. Эффект неодинаков в разные периоды.')
    st.info('Пересчёт после получения новых фактов анализируется отдельно и не подменяет конкурсную оценку прогноза, фиксируемого сразу на 13 недель.')

with tabs[3]:
    st.subheader('Проверка пользы сырьевых и межрыночных признаков')
    st.write('Сопоставляем модели одной семьи с внешними рядами и без них на одних исторических блоках. Внешние ряды — индикаторы из выданного датасета, а не дополнительные источники.')
    st.dataframe(pd.read_csv(ART/'ablation_metrics.csv'),hide_index=True,width='stretch')
    st.caption('Положительная improvement_pp — польза внешних признаков. Корреляция с ценой сама по себе не доказывает предсказательную силу или причинность.')
    if (ART/'feature_importance.csv').exists():
        importance=pd.read_csv(ART/'feature_importance.csv')
        if len(importance):
            st.subheader('Что использует финальная модель')
            st.bar_chart(importance.head(12).set_index('label').importance)
            st.caption('Важность признаков CatBoost описывает использование признаков моделью, а не причинный вклад в рыночную цену.')
    st.subheader('Как применять')
    st.write('Для продаж: оценивать сценарный диапазон будущей цены и пересматривать ориентиры после новых котировок. Для закупок: сопоставлять сценарии сырья и ПВХ, учитывая риск расхождения. Для планирования: показывать не только центральный прогноз, но и неопределённость на сроке решения.')
    st.subheader('Границы применимости')
    st.write('Ориентир DAP Достык не является готовой ценой каждого контракта. Анонимизированный индекс не позволяет вычислить прибыль в рублях. Контрактные сценарии с неоднозначными наблюдениями исключены; время публикации принимается равным дате записи. При смене режима рынка качество может ухудшиться.')
