"""Read-only demonstration of generated, reproducible artifacts."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parent/'artifacts'
st.set_page_config(page_title='Прогноз ПВХ', layout='wide')
st.title('Прогноз цены ПВХ · DAP Достык')
if not (ART/'run_metadata.json').exists():
    st.info('Сначала выполните: python src/pipeline.py')
    st.stop()
metadata = json.loads((ART/'run_metadata.json').read_text())
forecast = pd.read_csv(ART/'forecast_details.csv', parse_dates=['observation_date'])
history = pd.read_csv(ART/'target_history.csv', parse_dates=['report_date'])
st.caption('Анонимизированный показатель k. Горизонт — 13 недель. Обучение только на официальном файле.')
one, two, three = st.columns(3)
one.metric('Последнее наблюдение', metadata['target_last_date'])
two.metric('Модель', metadata['model'])
three.metric('Прогноз через 13 недель', f"{forecast.forecast.iloc[-1]:.4f}")
chart = pd.DataFrame(index=history.report_date.tail(104).tolist()+forecast.observation_date.tolist())
chart.loc[history.report_date.tail(104), 'История'] = history.k.tail(104).to_numpy()
for col, label in [('forecast','Прогноз'), ('lower','Нижняя граница'), ('upper','Верхняя граница')]:
    chart.loc[forecast.observation_date, label] = forecast[col].to_numpy()
st.line_chart(chart)
st.caption('Границы — эмпирические оценки по историческим ошибкам; покрытие на будущих данных не гарантировано.')
st.subheader('Прогноз и решения')
st.dataframe(forecast, hide_index=True, width='stretch')
st.caption('Позиция: +1 — ожидаемый рост, −1 — снижение, 0 — слабый сигнал. Учебный P&L не равен денежному эффекту компании.')
st.subheader('Историческая проверка')
st.write('Выбор модели: первые 6 временных блоков. Последние 2 блока оставлены для отдельной проверки.')
st.dataframe(pd.read_csv(ART/'leaderboard.csv'), hide_index=True)
outer = pd.read_csv(ART/'outer_evaluation.csv')
st.write(f"Проверено {outer.actual.notna().sum()} фактических недель во внешней оценке.")
coverage = outer.loc[outer.actual.notna(), 'covered'].mean()
st.warning(f'Покрытие предварительных интервалов на внешней проверке: {coverage:.1%}. Интервалы пока не откалиброваны для применения.')
st.subheader('Ограничения')
st.write('Даты публикации приняты за даты доступности. Неизвестные сроки публикации контрактных котировок — причина их исключения. Интервалы оценены на небольшом числе окон. Пороги стратегии требуют дальнейшей проверки.')
for filename in ['submission.csv', 'strategy.csv']:
    st.download_button(filename, (ART/filename).read_bytes(), file_name=filename, mime='text/csv')
