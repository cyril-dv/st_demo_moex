from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


rng = np.random.default_rng(seed=9637039)
dir_path = Path(__file__).parent.parent

@st.cache_data
def load_data():
    df = pd.read_csv(dir_path / 'data' / 'data.csv', parse_dates=['Date'], index_col='Date')
    return df

# Default values for inputs
stock_names = {
	'ALRS': 'Алроса',
	'CHMF': 'Северсталь',
	'GAZP': 'Газпром',
	'GMKN': 'Норильский никель',
	'IRAO': 'Интер РАО',
	'LKOH': 'Лукойл',
	'MGNT': 'Магнит',
	'MOEX': 'Московская Биржа',
	'MTSS': 'МТС',
	'NLMK': 'НЛМК',
	'NVTK': 'Новатэк',
	'PHOR': 'Фосагро',
	'PLZL': 'Полюс',
	'ROSN': 'Роснефть',
	'SBER': 'Сбербанк',
	'SNGSP': 'Сургутнефтегаз',
	'TATN': 'Татнефть',
	'TCSG': 'ТКС Холдинг',
	'VTBR': 'ВТБ',
	'YDEX': 'Яндекс'
}
stock_names_disp = dict(sorted(stock_names.items(), key=lambda x:x[1]))

if 'port_value' in st.session_state and 'share_stocks' in st.session_state:
    st.session_state['_port_stock_value'] = int(np.ceil(st.session_state['port_value'] * st.session_state['share_stocks'] / 100 / 10_000)*10_000)
else:
    st.session_state['_port_stock_value'] = 100_000

for ticker in stock_names.keys():
    if '_'+ticker not in st.session_state:
        st.session_state['_'+ticker] = 10

# Persistent session state
persist_widgets = ['port_stock_value'] + list(stock_names.keys())

def update_stock_weights():
    for ticker in stock_names.keys():
        st.session_state[ticker] = st.session_state['_'+ticker]
    
def store_widget_value(key):
    st.session_state[key] = st.session_state['_'+key]

def load_widget_value(key):
    if key in st.session_state:
        st.session_state['_'+key] = st.session_state[key]

for w in persist_widgets:
    load_widget_value(w)

st.header('Портфель из отдельных акций', divider='gray')
st.subheader('Исходные данные')
rename_dict = {
    'Date': 'Дата', 
    'IMOEX':'Индекс', 
    'IMOEX_DIV':'Акции', 
    'IMOEX_BNDS': 'Облигации',
    'PORT': 'Портфель'
}

df = load_data()
df = df.iloc[:, 0:-2].reset_index().rename(columns=rename_dict).set_index('Дата')

with st.expander('Данные в табличном виде'):
    st.dataframe(df, column_config={'Дата': st.column_config.DatetimeColumn(format='DD.MM.YYYY')})

selected_stock = st.pills('Обзор акций', df.columns[0:-1], selection_mode='single', default='SBER')

df_tmp = df.copy()
df_tmp = df_tmp[[selected_stock, 'Индекс']].reset_index()
df_tmp[selected_stock+'_'] = df_tmp[selected_stock]
df_tmp[selected_stock] = df_tmp[selected_stock] / df_tmp[selected_stock].iloc[0] * 100
df_tmp[selected_stock] = df_tmp[selected_stock].round(2)
df_tmp['Индекс'] = df_tmp['Индекс'] / df_tmp['Индекс'].iloc[0] * 100
df_tmp['Индекс'] = df_tmp['Индекс'].round(2)

col1, col2 = st.columns(2, gap='small')
fig = px.line(df_tmp, x='Дата', y=[selected_stock+'_'], hover_data={'Дата': '|%d.%m.%Y'}, template='seaborn')
fig.update_yaxes(title=selected_stock)
fig.update_xaxes(title=None, dtick='M6', tick0=df_tmp['Дата'].min(), tickformat='%b\n%Y')
fig.update_layout(legend=dict(title_text=None, orientation='h', x=0, y=1.1, xanchor='left', yanchor='top'))
col1.plotly_chart(fig)

fig = px.line(df_tmp, x='Дата', y=[selected_stock, 'Индекс'], hover_data={'Дата': '|%d.%m.%Y'}, template='seaborn')
fig.update_yaxes(title=selected_stock)
fig.update_xaxes(title=None, dtick='M6', tick0=df_tmp['Дата'].min(), tickformat='%b\n%Y')
fig.update_layout(legend=dict(title_text=None, orientation='h', x=0, y=1.1, xanchor='left', yanchor='top'))
col2.plotly_chart(fig)


st.subheader('Состав портфеля')
col1, col2 = st.columns(2, gap='large')
col1.number_input(label='Объем портфеля', min_value=10_000, max_value=10_000_000, step=100_000, value=st.session_state['_port_stock_value'], key='_port_stock_value', on_change=store_widget_value, args=['port_stock_value'])


with st.form('stock_weights', enter_to_submit=False):
    st.markdown('Доля акций')
    col1, col2, col3, col4 = st.columns(4, gap='large')
    with col1:
        for ticker, ticker_name in zip([k for k in stock_names_disp.keys()][0::4], [v for v in stock_names_disp.values()][0::4]):
            st.html(f"{ticker_name} <span style='color: #808080; font-size: 12px;'>{ticker}</span>")
            st.slider(ticker, label_visibility='collapsed', min_value=0, max_value=40, value=st.session_state['_'+ticker], step=1, key='_'+ticker)

    with col2:
        for ticker, ticker_name in zip([k for k in stock_names_disp.keys()][1::4], [v for v in stock_names_disp.values()][1::4]):
            st.html(f"{ticker_name} <span style='color: #808080; font-size: 12px;'>{ticker}</span>")
            st.slider(ticker, label_visibility='collapsed', min_value=0, max_value=40, value=st.session_state['_'+ticker], step=1, key='_'+ticker)

    with col3:
        for ticker, ticker_name in zip([k for k in stock_names_disp.keys()][2::4], [v for v in stock_names_disp.values()][2::4]):
            st.html(f"{ticker_name} <span style='color: #808080; font-size: 12px;'>{ticker}</span>")
            st.slider(ticker, label_visibility='collapsed', min_value=0, max_value=40, value=st.session_state['_'+ticker], step=1, key='_'+ticker)

    with col4:
        for ticker, ticker_name in zip([k for k in stock_names_disp.keys()][3::4], [v for v in stock_names_disp.values()][3::4]):
            st.html(f"{ticker_name} <span style='color: #808080; font-size: 12px;'>{ticker}</span>")
            st.slider(ticker, label_visibility='collapsed', min_value=0, max_value=40, value=st.session_state['_'+ticker], step=1, key='_'+ticker)

    weights_form = st.form_submit_button('Сохранить', on_click=update_stock_weights)

update_stock_weights()
st.subheader('Доли акций')

stock_weights = []
for ticker in stock_names.keys():
    stock_weights.append(st.session_state[ticker])
stock_weights = np.array(stock_weights)
stock_weights_norm = stock_weights / np.sum(stock_weights) * 100

df_stock_weights = pd.DataFrame({
        'Эмитент': stock_names.values(),
        'Тикер': stock_names.keys(),
        'Выбранные доли': stock_weights,
        'Нормализированные доли': stock_weights_norm
    }
).sort_values(by='Эмитент')
df_stock_weights = pd.concat([
        df_stock_weights,
        pd.DataFrame({
                'Эмитент': 'Итого',
                'Тикер': '',
                'Выбранные доли': np.sum(stock_weights),
                'Нормализированные доли': np.sum(stock_weights_norm)
            }, index=[1]
        )
])
df_stock_weights.index = range(1, df_stock_weights.shape[0] + 1)
st.dataframe(df_stock_weights, column_config={
                    'Выбранные доли': st.column_config.NumberColumn(format='%.2f'),
                    'Нормализированные доли': st.column_config.NumberColumn(format='%.2f')
            }
        )


st.subheader('Доходность')
# Yearly periods
CURRENT_YEAR = df.index.max().year

date_offsets = (
    [[df.index[-1] - pd.DateOffset(month=1, day=1), df.index[-1]]] +
    [[pd.Timestamp(i, 1, 1), pd.Timestamp(i, 12, 31)] for i in range(CURRENT_YEAR - 1, CURRENT_YEAR - 6, -1)] +
    [[pd.Timestamp(CURRENT_YEAR - 5, 1, 1), df.index[-1]]]
)
all_days = (df.index[-1] - pd.Timestamp(CURRENT_YEAR - 5, 1, 1)).days

# Portfolio returns based on selected weights
port_weights = stock_weights_norm / 100
df_returns = np.log(1 + df.pct_change()).dropna(how='all')
df_returns_disc = df.pct_change().dropna(how='all')
port_returns = np.sum(port_weights * df_returns_disc.iloc[:, 0:-1], axis=1)
port_returns = np.log(1 + port_returns)
df_returns['PORT'] = port_returns
df['PORT'] = np.r_[100, 100 * np.exp(np.cumsum(port_returns))]
df = df.rename(columns=rename_dict)
df = df.drop(columns=[i for i in df.columns if len(i) <= 5])
df_returns = df_returns.rename(columns=rename_dict)
df_returns = df_returns.drop(columns=[i for i in df_returns.columns if len(i) <= 5])

returns_ann = np.zeros((df.shape[1], len(date_offsets)))
for i, ticker in enumerate(df.columns):
    for j, tf in enumerate(date_offsets):
        ts = df.loc[tf[0]:tf[1], ticker]
        if tf == date_offsets[-1]:
            returns_ann[i, j] = (1 + ((ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0])) ** (365 / all_days) - 1
        else:
            returns_ann[i, j] = (ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0]
returns_ann *= 100
returns_ann = pd.DataFrame(
    returns_ann, 
    index=df.columns, 
    columns=[str(i) for i in range(CURRENT_YEAR, CURRENT_YEAR - 6, -1)] + ['В год. выражении']
)
st.dataframe(returns_ann, column_config={k:st.column_config.NumberColumn(format='%.2f') for k in returns_ann.columns})


st.subheader('Риск')
st.markdown('Волатильность')
vol_ann = np.zeros((df.shape[1], len(date_offsets)))
for i, ticker in enumerate(df.columns):
    for j, tf in enumerate(date_offsets):
        ts = df_returns.loc[tf[0]:tf[1], ticker]
        vol_ann[i, j] = ts.std() * np.sqrt(252)
vol_ann *= 100
vol_ann = pd.DataFrame(
    vol_ann, 
    index=df.columns, 
    columns=[str(i) for i in range(CURRENT_YEAR, CURRENT_YEAR - 6, -1)] + ['Весь период']
)
st.dataframe(vol_ann, column_config={k:st.column_config.NumberColumn(format='%.2f') for k in vol_ann.columns})


st.markdown('Максимальная просадка')
drawdawn = (df - df.cummax()) / df.cummax()
max_dd = np.zeros((df.shape[1], len(date_offsets)))
for i, ticker in enumerate(df.columns):
    for j, tf in enumerate(date_offsets):
        ts = drawdawn.loc[tf[0]:tf[1], ticker]
        max_dd[i, j] = ts.min()
max_dd *= 100
max_dd = pd.DataFrame(
    max_dd, 
    index=df.columns, 
    columns=[str(i) for i in range(CURRENT_YEAR, CURRENT_YEAR - 6, -1)] + ['Весь период']
)
st.dataframe(max_dd, column_config={k:st.column_config.NumberColumn(format='%.2f') for k in max_dd.columns})


st.markdown('Стоимость под риском (95% VaR и ES) на один месяц')
var_month = np.zeros((df.shape[1], 4))
for i, col in enumerate(df_returns):
    sim_returns = rng.choice(df_returns[col], size=(1_000_000, 21), replace=True, shuffle=True)
    sim_returns_terminal = np.prod(np.exp(sim_returns), axis=1)
    sim_returns_change = sim_returns_terminal - 1

    VaR95_pct = np.percentile(sim_returns_change, 5, method='inverted_cdf')
    VaR95_rub = VaR95_pct*st.session_state['_port_stock_value']
    ES95_pct = np.mean(sim_returns_change[sim_returns_change <= VaR95_pct])
    ES95_rub = ES95_pct*st.session_state['_port_stock_value']

    var_month[i, :] = np.array([VaR95_pct*100, VaR95_rub, ES95_pct*100, ES95_rub])

var_month = pd.DataFrame(
    var_month, 
    index=df.columns, 
    columns=['VaR, %', 'VaR, ₽', 'ES, %', 'ES, ₽']
)

st.dataframe(var_month.style.format({
                'VaR, %': lambda x: '{:,.2f}'.format(x),
                'VaR, ₽': lambda x: '{:,.0f}'.format(x),
                'ES, %': lambda x: '{:,.2f}'.format(x),
                'ES, ₽': lambda x: '{:,.0f}'.format(x),
            },
        thousands=' ',
        decimal='.',
    )
)


with st.expander('*Примечания*'):
    st.markdown(r'''
        [1] **Данные** за последние пять полных лет и за истекший период текущего года (01.01.2019 - 31.10.2024).  
        [2] Данные по тикеру **TCSG** за 10 месяцев 2019 года интерполированы из квартальных данных стоимости депозитарных расписок эмитента на LSE.  
        [3] **Акции** представлены Индексом МосБиржи ([Источник](https://www.moex.com/ru/index/IMOEX)).  
        [4] **Доходность** в годовом выражении рассчитана как $(1+R_{cumulative})^{{365} \over {days}}-1)$.  
        [5] Дневная **волатильность** пересчитана в годовую как $\sigma_{annualized} = \sigma_{daily}*\sqrt{252}$.  
        [6] Показатели **VaR** и **ES** получены с помощью исторического метода с бутстрапом.
    ''')
