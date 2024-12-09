import shutil
import streamlit as st
import pandas as pd
from streamlit_module import *
import os

tmp = 'tmp'
file_name = [file for file in os.listdir(tmp) if file.endswith('csv')][-1]
data = pd.read_csv(os.path.join(tmp,file_name))

st.markdown("<h1 style='text-align: center; font-size: 33px;'>Empower Your Trades with Technicals Insights</h1>", unsafe_allow_html=True)
st.markdown(
"<p style='text-align: center; font-size: 16px; font-weight: normal;'>Take control of your trading strategy with our advanced stock analysis platform. Easily identify top gainers, top losers, and actionable opportunities using AI-driven data. Leverage the 'Quick Filter' to streamline your search for the best stocks, whether you're seeking breakouts, uptrends, or RSI-based insights. Your journey to smarter, faster trading starts here!</p>",
unsafe_allow_html=True
)

with st.container(border=True):
    search_by_ticker(data)


st.markdown(
    """
    <style>
    .stRadio {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .stRadio > div {
        display: flex;
        flex-direction: row;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
options = ['All','Large cap', 'Mid Cap', 'Small Cap', 'Below 500(cr)', 'Above 500(cr)']
cap_size = st.radio("Filter Based On Market Size..", options, horizontal=True)

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

st.markdown("<h1 style='text-align: center;'>Quick Filter..</h1>", unsafe_allow_html=True)
with st.container(border=True):
    check_col1, check_col2, check_col3 = st.columns(3)
    with check_col1:
        low_rsi_positive = st.checkbox('LOW RSI VALUES')
        high_rel_vol =   st.checkbox('HIGH RELATIVE VOLUME')
        high_rsi_all =  st.checkbox('HIGH RSI VALUES')
        high_atr_vals = st.checkbox('HIGH ATR VALUES')
        high_momentum = st.checkbox('HIGH MOMENTUM')
        high_vol_price_corr = st.checkbox('HIGH CORRELATION BETWEEN VOLUME AND PRICE')

    with check_col2:
        breakout_high_vol = st.checkbox('BREAKOUT')
        breakout_consolidation = st.checkbox('BREAKOUT FROM CONSOLIDATION')
        super_trend =  st.checkbox('UP TREND')
        chart_ind = st.checkbox('LAST HIGH PRICE EXCEEDS')
        last_three_pos = st.checkbox('LAST THREE POSITIVE CANDELS')

    with check_col3:
        price_gap_pctg = st.checkbox('PRICE GAP UP')
        price_gap_consolidation = st.checkbox('PRICE GAP UP FROM CONSOLIDATION')
        nad_buy_signals = st.checkbox('CHART INDICATOR BUY SIGNALS ')
        nad_sell_signals = st.checkbox('CHART INDICATOR SELL SIGNALS ')
        candle_score = st.checkbox('CANDLE SCORE')

############################################################################################################
if low_rsi_positive:
    low_rsi_pos = data[(data['RSI'] <= 35) & (data['PRICE_DIFF'] > 0)].sort_values('RSI',ascending=False).reset_index(drop=True)
    low_rsi_pos = low_rsi_pos[['STOCK','PRICE', 'PRICE_DIFF']]
    low_rsi_pos = [list(v.values) for i, v in low_rsi_pos.iterrows()]
    display_matrix(low_rsi_pos, 'LOW RSI VALUES(+)')

if high_rsi_all:
    high_rsi_all = data[(data['RSI'] >= 70) & (data['PRICE_DIFF'] > 0)].sort_values('RSI',ascending=False).reset_index(drop=True)
    high_rsi_all = high_rsi_all[['STOCK','PRICE', 'PRICE_DIFF']]
    high_rsi_all = [list(v.values) for i, v in high_rsi_all.iterrows()]
    display_matrix(high_rsi_all, 'HIGH RSI VALUES')

if high_rel_vol:
    high_rel_vol = data[data['PRICE_DIFF'] > 0].sort_values('RELATIVE_VOLUME',ascending=False).reset_index(drop=True)
    high_rel_vol = high_rel_vol[['STOCK','PRICE', 'PRICE_DIFF']]
    high_rel_vol = [list(v.values) for i, v in high_rel_vol.iterrows()]
    display_matrix(high_rel_vol, 'HIGH RELATIVE VOLUME')

if high_atr_vals:
    high_atr_vals = data[(data['ATR'] >= 100)].sort_values('ATR',ascending=False).reset_index(drop=True)
    high_atr_vals = high_atr_vals[['STOCK','PRICE', 'PRICE_DIFF']]
    high_atr_vals = [list(v.values) for i, v in high_atr_vals.iterrows()]
    display_matrix(high_atr_vals, 'HIGH ATR VALUES ALL')

if high_momentum:
    high_momentum = data[(data['STRENGTH'] >= 0.9) & (data['TREND_THREE'] == 'INCREASING')].sort_values('STRENGTH',ascending=False).reset_index(drop=True)
    high_momentum = high_momentum[['STOCK','PRICE', 'PRICE_DIFF']]
    high_momentum = [list(v.values) for i, v in high_momentum.iterrows()]
    display_matrix(high_momentum, 'HIGH MOMENTUM')

if high_vol_price_corr:
    high_vol_price_corr = data[data['PRICE_DIFF'] > 0].sort_values('VOL_PRC_CORR',ascending=False).reset_index(drop=True)
    high_vol_price_corr = high_vol_price_corr[['STOCK','PRICE', 'PRICE_DIFF']]
    high_vol_price_corr = [list(v.values) for i, v in high_vol_price_corr.iterrows()]
    display_matrix(high_vol_price_corr, 'HIGH CORRELATION BETWEEN VOLUME AND PRICE')

###################################################################################################################
if breakout_high_vol:
    with st.container():
        brkout_high_vol = data[(data['BREAKOUT'] == True)& (data['PRICE_DIFF'] > 0)].sort_values('RELATIVE_VOLUME',ascending=False).reset_index(drop=True)
        brkout_high_vol = brkout_high_vol[['STOCK','PRICE', 'PRICE_DIFF']]
        brkout_high_vol = [list(v.values) for i, v in brkout_high_vol.iterrows()]
        display_matrix(brkout_high_vol, 'BREAKOUT')

if breakout_consolidation:
    with st.container():
        brkout_no_trend = data[(data['BREAKOUT'] == True)& (data['TREND_THREE'] == 'NO TREND') & (data['PRICE_DIFF'] > 0)].sort_values('RELATIVE_VOLUME',ascending=False).reset_index(drop=True)
        brkout_no_trend = brkout_no_trend[['STOCK','PRICE', 'PRICE_DIFF']]
        brkout_no_trend = [list(v.values) for i, v in brkout_no_trend.iterrows()]
        display_matrix(brkout_no_trend, 'BREAKOUT FROM CONSOLIDATION')

if super_trend:
    with st.container():
        up_trend = data[(data['UPTREND_INDICATOR'] == 'STRONG BUY')& (data['PRICE_DIFF'] > 0)].sort_values('RSI',ascending=True).reset_index(drop=True)
        up_trend = up_trend[['STOCK','PRICE', 'PRICE_DIFF']]
        up_trend = [list(v.values) for i, v in up_trend.iterrows()]
        display_matrix(up_trend, 'SUPER TREND')

if chart_ind:
    chat_indctor = data[(data['CHART_INDICATOR'] == 'STRONG BUY')& (data['PRICE_DIFF'] > 0)].sort_values('RSI',ascending=True).reset_index(drop=True)
    chat_indctor = chat_indctor[['STOCK','PRICE', 'PRICE_DIFF']]
    chat_indctor = [list(v.values) for i, v in chat_indctor.iterrows()]
    display_matrix(chat_indctor, 'LAST HIGH PRICE EXCEEDS')

if last_three_pos:
    last_three_pos = data[(data['LAST_THREE_CANDEL'] == True)& (data['PRICE_DIFF'] > 0)].sort_values('CDL_SCR_TDY',ascending=False).reset_index(drop=True)
    last_three_pos = last_three_pos[['STOCK','PRICE', 'PRICE_DIFF']]
    last_three_pos = [list(v.values) for i, v in last_three_pos.iterrows()]
    display_matrix(last_three_pos, 'LAST THREE POSITIVE CANDELS')

###########################################################################################################
if price_gap_pctg:
    with st.container():
        pos_gap_high_vol = data[(data['PEICE_GAP_PCTG'] > 0)& (data['PRICE_DIFF'] > 0)].sort_values('PEICE_GAP_PCTG',ascending=False).reset_index(drop=True)
        pos_gap_high_vol = pos_gap_high_vol[['STOCK','PRICE', 'PEICE_GAP_PCTG']]
        pos_gap_high_vol = [list(v.values) for i, v in pos_gap_high_vol.iterrows()]

        neg_gap_high_vol = data[(data['PEICE_GAP_PCTG'] < 0)& (data['PRICE_DIFF'] < 0)].sort_values('PEICE_GAP_PCTG',ascending=False).reset_index(drop=True)
        neg_gap_high_vol = neg_gap_high_vol[['STOCK','PRICE', 'PEICE_GAP_PCTG']]
        neg_gap_high_vol = [list(v.values) for i, v in neg_gap_high_vol.iterrows()][::-1]

        price_gap = pos_gap_high_vol+neg_gap_high_vol
        display_matrix(price_gap, 'PRICE GAP PERCENTAGE')

if price_gap_consolidation:
    price_gap_consolidation = data[(data['PEICE_GAP_PCTG'] > 0)& (data['TREND_THREE'] == 'NO TREND')& (data['PRICE_DIFF'] > 0)].sort_values('RELATIVE_VOLUME',ascending=False).reset_index(drop=True)
    price_gap_consolidation = price_gap_consolidation[['STOCK','PRICE', 'PRICE_DIFF']]
    price_gap_consolidation = [list(v.values) for i, v in price_gap_consolidation.iterrows()]
    display_matrix(price_gap_consolidation, 'GAP UP FROM CONSOLIDATION')

if nad_buy_signals:
    nad_buy_signals = data[data['NADARAYA_WATSON']=='BUY'].sort_values(['RELATIVE_VOLUME','RSI'], ascending = False).reset_index(drop=True)
    nad_buy_signals = nad_buy_signals[['STOCK','PRICE', 'PRICE_DIFF']]
    nad_buy_signals = [list(v.values) for i, v in nad_buy_signals.iterrows()]
    display_matrix(nad_buy_signals, 'CHART INDICATOR BUY SIGNALS')

if nad_sell_signals:
    nad_sell_signals = data[data['NADARAYA_WATSON']=='SELL'].sort_values(['RELATIVE_VOLUME','RSI'], ascending = False).reset_index(drop=True)
    nad_sell_signals = nad_sell_signals[['STOCK','PRICE', 'PRICE_DIFF']]
    nad_sell_signals = [list(v.values) for i, v in nad_sell_signals.iterrows()]
    display_matrix(nad_sell_signals, 'CHART INDICATOR SELL SIGNALS')

if candle_score:
    candle_score = data[data['CDL_SCR_TDY'] > 0].sort_values('CDL_SCR_TDY', ascending = False).reset_index(drop=True)
    candle_score = candle_score[['STOCK','PRICE', 'PRICE_DIFF']]
    candle_score = [list(v.values) for i, v in candle_score.iterrows()]
    display_matrix(candle_score, 'CANDLE SCORE')