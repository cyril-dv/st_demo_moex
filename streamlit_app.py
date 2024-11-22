from pathlib import Path

import streamlit as st


dir_path = Path(__file__).parent

st.set_page_config(
    page_title='Доходность и риск ценных бумаг на Мосбирже',
    page_icon=':abacus:',
    layout='wide',
    initial_sidebar_state='expanded'
)

page = st.navigation([
    st.Page(dir_path / 'sections' / 'desc.py', title='Описание', icon=':material/page_info:'),
    st.Page(dir_path / 'sections' / 'stocks_bonds.py', title='Портфель акций и облигаций', icon=':material/finance_mode:'),
    st.Page(dir_path / 'sections' / 'stocks_port.py', title='Портфель из отдельных акций', icon=':material/bar_chart:')
])
page.run()
