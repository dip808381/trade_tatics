import shutil
import time
import streamlit as st
import pandas as pd
from streamlit_module import *
import os
st.markdown("<h1 style='text-align: center;'>WELCOME TO TRADE TACTICS</h1>", unsafe_allow_html=True)

tmp = 'tmp'
forecast_json_path = 'daily_forecast.json'

if not os.path.exists(tmp) or len(os.listdir(tmp)) < 2:
    try:
        shutil.rmtree(tmp)
        os.mkdir(tmp)
    except:
        os.mkdir(tmp)
    mod_date_time, csv_data, json_pred = load_data()
    mod_date, mod_time = extract_date_and_time(mod_date_time)
    mod_time = mod_time.replace(":", "_")
    f_name = f"{mod_date}_{mod_time}.csv"
    csv_data.to_csv(os.path.join(tmp,f_name), index=False)
    dump_json(os.path.join(tmp,forecast_json_path), json_pred)

file_name = [file for file in os.listdir(tmp) if file.endswith('csv')][-1]
data = pd.read_csv(os.path.join(tmp,file_name))

forecasted_json = read_json(os.path.join(tmp,forecast_json_path))
forecasted_df = pd.DataFrame(forecasted_json[list(forecasted_json.keys())[-1]], columns=['TICKER', 'COMPANY_NAME', 'MARKET_CAP', 'PRICE', 'RATE(%)','PRED_PRICE'])
forecasted_df = forecasted_df.round(2)

# data refreshed
current_date, current_time = get_current_date_time()
mod_date, mod_time = extract_date_time_from_filename(file_name)

data = data[['STOCK', 'COMPANY_NAME', 'MARKET_CAP(CR)', 'VOLUME', 'RELATIVE_VOLUME', 'PRICE',
       'TREND_THREE', 'TREND_SIX', 'RSI', 'ATR',
       'RSI_INDICATOR', 'UPTREND_INDICATOR', 'MOVING_AVG_IND', 'STRENGTH',
       'CHART_INDICATOR', 'BREAKOUT', 'LAST_THREE_CANDEL',
       'VOL_PRC_CORR', 'PRICE_DIFF', 'PEICE_GAP_PCTG','NADARAYA_WATSON', 'CDL_NME_TDY', 'CDL_SCR_TDY']]

with st.container(border=True):
    note, update = st.columns(2)
    with note:
        #st.write("Press refresh if last updated date doesn't match today's date.")
        refresh = st.button('Refresh', type="primary")
        if refresh:
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            shutil.rmtree('tmp')
            tmp = 'tmp'
            if not os.path.exists(tmp):
                os.mkdir(tmp)
                mod_date_time, csv_data, json_pred = load_data()
                mod_date, mod_time = extract_date_and_time(mod_date_time)
                mod_time = mod_time.replace(":", "_")
                f_name = f"{mod_date}_{mod_time}.csv"
                csv_data.to_csv(os.path.join(tmp,f_name), index=False)
                dump_json(os.path.join(tmp,forecast_json_path), json_pred)
            my_bar.empty()

    with update:
        if current_date == mod_date:
            st.success('As on - ' + mod_date +" "+mod_time.replace("_", ":"))
        else:
            st.warning('As on - ' + mod_date +" "+mod_time.replace("_", ":"))

    # Initialize session state variable to detect refresh
    if "page_refreshed" not in st.session_state:
        st.session_state.page_refreshed = True  
    # Code to run only on page refresh
    if st.session_state.page_refreshed:
        market_data = display_market()
        display_matrix(market_data, 'Indian Indices',columns_per_row = 4, percentage=False)

with st.container(border=True):
    search_by_ticker(data)

with st.container(border=True):
    st.markdown("<h1 style='text-align: center; font-size: 24px;'>AI-Powered Stock Predictions</h1>", unsafe_allow_html=True)
    st.markdown(
    "<p style='text-align: center; font-size: 16px; font-weight: normal;'>Discover the future of investing with our AI-driven forecast models. This table highlights top-performing stocks, their predicted prices, and insights to help you make informed decisions.</p>",
    unsafe_allow_html=True
)
    tog_on = st.toggle(f'View all')
    if tog_on:
        try:
            st.dataframe(forecasted_df[:20], hide_index=True,use_container_width=True)
        except:
            st.dataframe(forecasted_df, hide_index=True,use_container_width=True)
    else:
        st.dataframe(forecasted_df[:10], hide_index=True,use_container_width=True)

cap_size = st.selectbox('FILTER BASED ON MARKET SIZE', ['All','Large cap', 'Mid Cap', 'Small Cap', 'Below 500(cr)', 'Above 500(cr)'])
if cap_size == 'All':
    data = data
if cap_size == 'Large cap':
    data = data[data['MARKET_CAP(CR)'] >= 20000].reset_index(drop=True)
if cap_size == 'Mid Cap':
    data = data[data['MARKET_CAP(CR)'].between(5000, 20000)].reset_index(drop=True)
if cap_size == 'Small Cap':
    data = data[data['MARKET_CAP(CR)'] <= 5000].reset_index(drop=True)
if cap_size == 'Below 500(cr)':
    data = data[data['MARKET_CAP(CR)'] <= 500].reset_index(drop=True)
if cap_size == 'Above 500(cr)':
    data = data[data['MARKET_CAP(CR)'] >= 500].reset_index(drop=True)

with st.container():
    gainers_losers_df= data[['STOCK','PRICE', 'PRICE_DIFF']].sort_values(by='PRICE_DIFF', ascending= False).reset_index(drop=True)
    gainers = [list(v.values) for i, v in gainers_losers_df.iterrows() if v.values[2] > 5]
    display_matrix(gainers, 'Top Gainers')

    losers = [list(v.values) for i, v in gainers_losers_df.iterrows() if v.values[2] < -5][::-1]
    display_matrix(losers, 'Top Losers')
