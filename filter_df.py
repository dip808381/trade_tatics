import os
import streamlit as st
from streamlit_module import *
import pandas as pd
import numpy as np
from datetime import date, timedelta
import shutil

tmp = 'tmp'

file_name = [file for file in os.listdir(tmp) if file.endswith('csv')][-1]
data = pd.read_csv(os.path.join(tmp,file_name))

current_date, current_time = get_current_date_time()
mod_date, mod_time = extract_date_time_from_filename(file_name)

hed1, hed2 = st.columns(2)

with hed1:
    st.title("Filter Stocks ")
with hed2:
    if current_date == mod_date:
            st.success('As on - ' + mod_date +" "+mod_time.replace("_", ":"))
    else:
        st.warning('As on - ' + mod_date +" "+mod_time.replace("_", ":"))
# st.write(
#     """This app accomodates the blog [here](https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/)
#     and walks you through one example of how the Streamlit
#     Data Science Team builds add-on functions to Streamlit.
#     """
# )
st.write('With this app, you get access to comprehensive technical analysis using the latest data sets. Easily filter to find the best companies that fit your criteria and start investing today!')

 
data = data[['STOCK', 'COMPANY_NAME', 'MARKET_CAP(CR)', 'VOLUME', 'RELATIVE_VOLUME', 'PRICE',
       'TREND_THREE', 'TREND_SIX', 'RSI', 'ATR',
       'RSI_INDICATOR', 'UPTREND_INDICATOR', 'MOVING_AVG_IND', 'STRENGTH',
       'CHART_INDICATOR', 'BREAKOUT', 'LAST_THREE_CANDEL',
       'VOL_PRC_CORR', 'PRICE_DIFF', 'PEICE_GAP_PCTG', 'CDL_NME_TDY', 'CDL_SCR_TDY']]

col_name_change = rename_columns()

with st.container(border=True):
    search_by_ticker(data)

# filter_toggle = st.toggle('Click here if you want to add filters..')
# if filter_toggle:
with st.container(border=True):
    data, selective_df, all_selected = filter_stocks(data, col_name_change)
    if len(all_selected.strip()) != 0:
        st.success(f'You are searching for {all_selected}...'.upper())
    else:
        st.warning('Select columns to filter the best stocks...')
    selective_df_toggle = st.toggle('Click here if you want to see only selective columns..')
    if len(selective_df.columns) > 4 and selective_df_toggle:
        st.text("""Market capital, volume and price are included in selective column search""")
        st.dataframe(selective_df.reset_index(drop=True), height = 600, width=1500)
    else:
        st.dataframe(data.reset_index(drop=True), height = 600, width=1500)
# else:
#     st.dataframe(data.reset_index(drop=True), height = 600, width=1500)

