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
if '_port_value' not in st.session_state:
    st.session_state['_port_value'] = 100_000
if '_share_stocks' not in st.session_state:
    st.session_state['_share_stocks'] = 50
if '_share_bonds' not in st.session_state:
    st.session_state['_share_bonds'] = 50

# Persistent session state
persist_widgets = ['port_value']

def update_share_stocks():
    st.session_state['_share_stocks'] = 100 - st.session_state['_share_bonds']
    st.session_state['share_stocks'] = st.session_state['_share_stocks']

def update_share_bonds():
    st.session_state['_share_bonds'] = 100 - st.session_state['_share_stocks']
    st.session_state['share_stocks'] = st.session_state['_share_stocks']

def store_widget_value(key):
    st.session_state[key] = st.session_state['_'+key]

def load_widget_value(key):
    if key in st.session_state:
        st.session_state['_'+key] = st.session_state[key]

for w in persist_widgets:
    load_widget_value(w)


st.header('Портфель акций и облигаций', divider='gray')
st.subheader('Исходные данные')
rename_dict = {
    'Date': 'Дата', 
    'IMOEX':'Индекс', 
    'IMOEX_DIV':'Акции', 
    'IMOEX_BNDS': 'Облигации',
    'PORT': 'Портфель'
}

df = load_data()
df = df[['IMOEX_DIV', 'IMOEX_BNDS']]
df_tmp = df.copy().reset_index().rename(columns=rename_dict)

with st.expander('Данные в табличном виде'):
    st.dataframe(df_tmp, width=400, hide_index=True, column_config={
                'Дата': st.column_config.DatetimeColumn(format='DD.MM.YYYY'),
                'Акции': st.column_config.NumberColumn(format='%.2f'),
                'Облигации': st.column_config.NumberColumn(format='%.2f')
            }
        )

df_tmp['Акции'] = df_tmp['Акции'] / df_tmp['Акции'].iloc[0] * 100
df_tmp['Акции'] = df_tmp['Акции'].round(2)
df_tmp['Облигации'] = df_tmp['Облигации'] / df_tmp['Облигации'].iloc[0] * 100
df_tmp['Облигации'] = df_tmp['Облигации'].round(2)

fig = px.line(df_tmp, x='Дата', y=['Акции', 'Облигации'], hover_data={'Дата': '|%d.%m.%Y'}, template='seaborn')
fig.update_yaxes(title=f'{df_tmp['Дата'].min().year}=100')
fig.update_xaxes(title=None, dtick='M6', tick0=df_tmp['Дата'].min(), tickformat='%b\n%Y')
fig.update_layout(legend=dict(title_text=None, orientation='h', x=0, y=1.1, xanchor='left', yanchor='top'))
st.plotly_chart(fig)


st.subheader('Состав портфеля')
col1, col2 = st.columns(2, gap='large')
col1.number_input(label='Объем портфеля', min_value=10_000, max_value=10_000_000, step=100_000, value= st.session_state['_port_value'], key='_port_value', on_change=store_widget_value, args=['port_value'])

col1, col2 = st.columns(2, gap='large')
if 'share_stocks' in st.session_state:
    st.session_state['_share_stocks'] = st.session_state['share_stocks']
    st.session_state['_share_bonds'] = 100 - st.session_state['share_stocks']
col1.slider('Доля акций', min_value=0, max_value=100, value=st.session_state['_share_stocks'], step=1, key='_share_stocks', on_change=update_share_bonds)
col2.slider('Доля облигаций', min_value=0, max_value=100, value=st.session_state['_share_bonds'], step=1, key='_share_bonds', on_change=update_share_stocks)


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
port_weights = np.array([st.session_state['_share_stocks'], 100-st.session_state['_share_stocks']]) / 100
df_returns = np.log(1 + df.pct_change()).dropna(how='all')
df_returns_disc = df.pct_change().dropna(how='all')
port_returns = np.sum(port_weights * df_returns_disc, axis=1)
port_returns = np.log(1 + port_returns)
df_returns['PORT'] = port_returns
df['PORT'] = np.r_[100, 100 * np.exp(np.cumsum(port_returns))]
df = df.reset_index().rename(columns=rename_dict).set_index('Дата')
df_returns = df_returns.reset_index().rename(columns=rename_dict).set_index('Дата')

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
    sim_returns = rng.choice(df_returns[col], size=(100_000, 21), replace=True, shuffle=True)
    sim_returns_terminal = np.prod(np.exp(sim_returns), axis=1)
    sim_returns_change = sim_returns_terminal - 1

    VaR95_pct = np.percentile(sim_returns_change, 5, method='inverted_cdf')
    VaR95_rub = VaR95_pct*st.session_state['_port_value']
    ES95_pct = np.mean(sim_returns_change[sim_returns_change <= VaR95_pct])
    ES95_rub = ES95_pct*st.session_state['_port_value']

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
        [2] **Акции** представлены Индексом МосБиржи полной доходности "нетто" по налоговым ставкам российских организаций ([Источник](https://www.moex.com/ru/index/totalreturn/MCFTR)).  
        [3] **Облигации** представлены Индексом МосБиржи корпоративных облигаций, учитывающим совокупный доход ([Источник](https://www.moex.com/ru/index/RUCBTRNS)).  
        [4] **Доходность** в годовом выражении рассчитана как $(1+R_{cumulative})^{{365} \over {days}}-1)$.  
        [5] Дневная **волатильность** пересчитана в годовую как $\sigma_{annualized} = \sigma_{daily}*\sqrt{252}$.  
        [6] Показатели **VaR** и **ES** получены с помощью исторического метода с бутстрапом.
    ''')
