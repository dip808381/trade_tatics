import re
import pytz
import requests
import json
import pandas as pd
import numpy as np
from io import StringIO
import streamlit as st
from datetime import datetime
from bs4 import BeautifulSoup
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from azure.storage.blob import BlobServiceClient
ist = pytz.timezone('Asia/Kolkata')
import os

def wide_space_default():
    st.set_page_config(layout='wide') 

def rename_columns():
    return {'STOCK':'TICKER', 'COMPANY_NAME':'COMPANY NAME', 'MARKET_CAP(CR)':'MARKET CAPITAL(CR)', 'VOLUME':'TOTAL VOLUME', 'RELATIVE_VOLUME':'RELATIVE VOLUME', 'PRICE':'CURRENT PRICE',
       'TREND_THREE':'LAST THREE MONTHS TREND', 'TREND_SIX':'LAST SIX MONTHS TREND', 'RSI':'RSI VALUE', 'ATR':'ATR VALUE',
       'RSI_INDICATOR':'RSI INDICATOR', 'UPTREND_INDICATOR':'UPTREND INDICATOR', 'MOVING_AVG_IND':'MOVING AVERAGE INDICATOR', 'STRENGTH':'STRENGTH',
       'CHART_INDICATOR':'LAST HIGH PRICE EXCEEDS', 'BREAKOUT':'BREAKOUT', 'LAST_THREE_CANDEL':'LAST THREE POSITIVE CANDLE',
       'VOL_PRC_CORR':'VOLUME PRICE CORRELATION', 'PRICE_DIFF':'PRICE DIFFERENCE', 'PEICE_GAP_PCTG':'PRICE GAP PERCENTAGE', 'CDL_NME_TDY':'TODAY CANDLESTICKS', 'CDL_SCR_TDY':'TODAY CANDLESTICK SCORE','NADARAYA_WATSON':'CHART INDICATOR',
       'CDL_NME_YES':'YESTERDAY CANDLESTICKS', 'CDL_SCR_YES':'YESTERDAY CANDLESTICK SCORE'}

def numeric_filter(df, column_name, col_name_change):

    numeric_options = ['Any..','Greater Than..', 'Less Than..','Equal..','Does Not Equal..', 'Between..']
    num_fil = st.selectbox(col_name_change[column_name], numeric_options)

    if num_fil == 'Greater Than..':
        value = st.number_input(f"Insert {column_name.lower()}",value=None, placeholder="Type a number...", step =1)
        ret_val = df[df[column_name]>=value]
        str_name = f"{column_name.lower()} is {num_fil[:-2]} {value}"
        return ret_val, str_name

    elif num_fil == 'Less Than..':
        value = st.number_input(f"Insert {column_name.lower()}",value=None, placeholder="Type a number...", step =1)
        ret_val = df[df[column_name]<=value]
        str_name = f"{column_name.lower()} is {num_fil[:-2]} {value}"
        return ret_val, str_name

    elif num_fil == 'Equal..':
        value = st.number_input(f"Insert {column_name.lower()}",value=None, placeholder="Type a number...", step =1)
        ret_val = df[df[column_name]==value]
        str_name = f"{column_name.lower()} is {num_fil[:-2]} {value}"
        return ret_val, str_name

    elif num_fil == 'Does Not Equal..':
        value = st.number_input(f"Insert {column_name.lower()}",value=None, placeholder="Type a number...", step =1)
        ret_val = df[df[column_name]!=value]
        str_name = f"{column_name.lower()} is {num_fil[:-2]} {value}"
        return ret_val, str_name
    
    elif num_fil =='Between..':
        min_, max_ = st.columns(2, gap='small')
        with min_:
            min_val = st.number_input(f"Insert min {column_name.lower()}",value=None, placeholder="Min value...", step =1,label_visibility ="collapsed")
        with max_:
            max_val = st.number_input(f"Insert max {column_name.lower()}",value=None, placeholder="Max value...", step =1,label_visibility ="collapsed")
        ret_val = df[df[column_name].between(min_val, max_val)]
        str_name = f"{column_name.lower()} range in between {min_val} and {max_val}"
        return ret_val, str_name
    
    else:
        return df, ''

def search_by_ticker(data):
    col_name_change = rename_columns()
    col_name_change = {key: value.title() for key, value in col_name_change.items()}
    # data.rename(columns=rename_columns(), inplace=True)
    ser_name = st.selectbox('GET ALL TECHNICAL AND FUNDAMENTAL ANALYSIS..', ['TICKER','COMPANY_NAME'], placeholder='search by company or ticker..')

    if ser_name == 'COMPANY_NAME':
        stk_comp = st.selectbox('Search by company name..', list(data['COMPANY_NAME'].unique()), index=None, placeholder='search by company..')
        if stk_comp:
            df_comp = data[data['COMPANY_NAME'].isin([stk_comp])]
            df_comp = df_comp.rename(columns=col_name_change)
            df_comp = df_comp.T
            df_comp.reset_index(inplace=True)
            df_comp.columns = ['COLUMNS', 'VALUES']
            stk_tic = df_comp.loc[df_comp['COLUMNS'] == 'Ticker']
            stk_tic_val = stk_tic['VALUES'][0]
            plot_candle(stk_tic_val)
            st.markdown("<h1 style='text-align: center; font-size: 24px;'>Technical Analysis</h1>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df_comp[:len(df_comp)//2],height=420,hide_index=True,use_container_width=True)
            with col2:
                st.dataframe(df_comp[len(df_comp)//2:],height=420,hide_index=True,use_container_width=True)
            st.markdown("<h1 style='text-align: center; font-size: 24px;'>Fundamental Analysis</h1>", unsafe_allow_html=True)
            try:
                df_fund, fund_changes = get_fundamentals(stk_tic_val)
                st.dataframe(df_fund, width=750)
                st.table(fund_changes)
                
            except:
                st.warning('Dataset is not available')

    elif ser_name == 'TICKER':
        stk_tic = st.selectbox('Search by ticker..', list(data['STOCK'].unique()), index=None, placeholder='search by ticker..')
        if stk_tic:
            df_tic = data[data['STOCK'].isin([stk_tic])]
            df_tic = df_tic.rename(columns=col_name_change)
            df_tic = df_tic.T
            df_tic.reset_index(inplace=True)
            df_tic.columns = ['COLUMNS', 'VALUES']
            plot_candle(stk_tic)
            st.markdown("<h1 style='text-align: center; font-size: 24px;'>Technical Analysis</h1>", unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                st.dataframe(df_tic[:len(df_tic)//2],height=420,hide_index=True,use_container_width=True)
            with col4:
                st.dataframe(df_tic[len(df_tic)//2:],height=420,hide_index=True,use_container_width=True)
            st.markdown("<h1 style='text-align: center; font-size: 24px;'>Fundamental Analysis</h1>", unsafe_allow_html=True)
            try:
                df_fund, fund_changes = get_fundamentals(stk_tic)
                st.dataframe(df_fund, width=750)
                st.table(fund_changes)
            except:
                st.warning('Dataset is not available')


def filter_stocks(data, col_name_change):
    c1, c2, c3 = st.columns(3, gap='small')
    with c1:
        trend_three = st.multiselect('THREE MONTHS TREND', list(data['TREND_THREE'].unique()))
        if trend_three:
            data = data[data['TREND_THREE'].isin(trend_three)]
            TREND_THREE = f"{'TREND_THREE'} is {''.join(trend_three)}"

        else:
            TREND_THREE = ''

        trend_six = st.multiselect('SIX MONTHS TREND', list(data['TREND_SIX'].unique()))
        if trend_six:
            data = data[data['TREND_SIX'].isin(trend_six)]
            TREND_SIX = f"{'TREND_SIX'} is {''.join(trend_six)}"
        
        else:
            TREND_SIX = ''

        rsi_ind = st.multiselect('RSI INDICATOR', list(data['RSI_INDICATOR'].unique()))
        if rsi_ind:
            data = data[data['RSI_INDICATOR'].isin(rsi_ind)]
            RSI_INDICATOR = f"{'RSI_INDICATOR'} is {''.join(rsi_ind)}"

        else:
            RSI_INDICATOR = ''

        up_trnd = st.multiselect('UP TREND', list(data['UPTREND_INDICATOR'].unique()))
        if up_trnd:
            data = data[data['UPTREND_INDICATOR'].isin(up_trnd)]
            UPTREND_INDICATOR = f"{'UPTREND_INDICATOR'} is {''.join(up_trnd)}"
        else:
            UPTREND_INDICATOR = ''

        mvng_avg = st.multiselect('MOVING AVERAGE', list(data['MOVING_AVG_IND'].unique()))
        if mvng_avg:
            data = data[data['MOVING_AVG_IND'].isin(mvng_avg)]
            MOVING_AVG_IND = f"{'MOVING_AVG_IND'} is {''.join(mvng_avg)}"
        else:
            MOVING_AVG_IND = ''

        chart_ind = st.multiselect('CHART INDICATOR', list(data['CHART_INDICATOR'].unique()))
        if chart_ind:
            data = data[data['CHART_INDICATOR'].isin(chart_ind)]
            CHART_INDICATOR = f"{'CHART_INDICATOR'} is {''.join(chart_ind)}"
        else:
            CHART_INDICATOR = ''

        cdl_name = st.multiselect('CNADLESTICKS', list(set([val for sublist in data['CDL_NME_TDY'] if type(sublist)== str for val in eval(sublist)])))
        if cdl_name:
            data = data.loc[data.apply(lambda row: row.astype(str).str.contains(*cdl_name).any(), axis=1)]
            CDL_NME_TDY = f"{'CDL_NME_TDY'} is {''.join(cdl_name)}"
        else:
            CDL_NME_TDY = ''

    with c2:
        data, MR_CAP = numeric_filter(data, 'MARKET_CAP(CR)', col_name_change)
        data, VOLUME = numeric_filter(data, 'VOLUME', col_name_change)
        data, REL_VOL = numeric_filter(data, 'RELATIVE_VOLUME', col_name_change)
        data, PRICE = numeric_filter(data, 'PRICE', col_name_change)
        data, RSI = numeric_filter(data, 'RSI', col_name_change)
        data, ATR = numeric_filter(data, 'ATR', col_name_change)

        brk_out = st.multiselect('BREAKOUT', list(data['BREAKOUT'].unique()))
        if brk_out:
            data = data[data['BREAKOUT'].isin(brk_out)]
            BREAKOUT = f"{'BREAKOUT'} is {','.join(str(v) for v in brk_out)}"
        else:
            BREAKOUT = ''

    with c3:
        data, PRICE_DIFF = numeric_filter(data, 'PRICE_DIFF', col_name_change)
        data, PEICE_GAP_PCTG = numeric_filter(data, 'PEICE_GAP_PCTG', col_name_change)
        data, VOL_PRC_CORR = numeric_filter(data, 'VOL_PRC_CORR', col_name_change)
        data, STRENGTH = numeric_filter(data, 'STRENGTH', col_name_change)
        data, CDL_SCR_TDY = numeric_filter(data, 'CDL_SCR_TDY', col_name_change)

        lst_three = st.multiselect('THREE POSITIVE CANDEL', list(data['LAST_THREE_CANDEL'].unique()))
        if lst_three:
            data = data[data['LAST_THREE_CANDEL'].isin(lst_three)]
            LAST_THREE_CANDEL = f"{'LAST_THREE_CANDEL'} is {','.join(str(v) for v in lst_three)}"
        else:
            LAST_THREE_CANDEL = ''
    
    all_cols = [TREND_THREE, TREND_SIX, RSI_INDICATOR, UPTREND_INDICATOR, MOVING_AVG_IND, CHART_INDICATOR, CDL_NME_TDY, BREAKOUT, MR_CAP, VOLUME, REL_VOL,
                    PRICE, RSI, ATR, PRICE_DIFF, PEICE_GAP_PCTG, VOL_PRC_CORR, STRENGTH, CDL_SCR_TDY, LAST_THREE_CANDEL]
    all_selected = '  '.join(all_cols)
    
    uniq_cols = ['STOCK', 'MARKET_CAP(CR)', 'VOLUME', 'PRICE']
    vars_as_strings = uniq_cols+[col.split()[0].upper() for col in all_cols if isinstance(col, str) and len(col) != 0 and col.split()[0].upper() not in uniq_cols]
    if len(vars_as_strings) != 0:
        selective_df = data[vars_as_strings]
        return data, selective_df, all_selected
    else:
        selective_df = ''
        return data, selective_df, all_selected

def display_matrix(values, header, columns_per_row = 6, percentage=True):
    head, tog = st.columns(2)
    with head:
        st.header(header)
    with tog:
        #tog_on = f'tog_on_{header.lower}'
        tog_on = st.toggle(f'View all {header.lower()}')

    if tog_on:
        # Loop through values and display them in rows of columns
        for i in range(0, len(values), columns_per_row):
            # Create a new row of columns
            cols = st.columns(columns_per_row)
            # Populate each column in the row
            for col, value in zip(cols, values[i:i + columns_per_row]):
                #col.write(value)
                if percentage:
                    col.metric(value[0], round(value[1]), str(round(value[2])) + '%')
                else:
                    col.metric(value[0], value[1], str(value[2])+ '%')


    else:
        values = values[:columns_per_row*2]
        # Loop through values and display them in rows of columns
        for i in range(0, len(values), columns_per_row):
            # Create a new row of columns
            cols = st.columns(columns_per_row)
            # Populate each column in the row
            for col, value in zip(cols, values[i:i + columns_per_row]):
                #col.write(value)
                if percentage:
                    col.metric(value[0], round(value[1]), str(round(value[2])) + '%')
                else:
                    col.metric(value[0], value[1], str(value[2])+ '%')


# fundamental analysis
def Merge(d1, d2):
    for i in d2.keys():
        d1[i]=d2[i]
    return d1

def growth(values):
    values = values[-3:]
    try:
        pect = ((values[-1] - values[0])/abs(values[0]))*100
    except:
        pect = 0
    return round(pect, 2)

def read_json(json_path):
    with open(json_path, 'r+', encoding='utf-8') as test:
        json_file = json.load(test)
    return json_file

def dump_json(json_path, dic):
    with open(json_path, 'w') as fp:
        json.dump(dic, fp, indent=4)

def remove_special_characters(text):
    # Define the pattern for special characters using regular expression
    pattern = r'[^a-zA-Z0-9\s]'  # This pattern matches anything that is not alphanumeric or whitespace

    # Use re.sub() to replace the special characters with an empty string
    text_without_special_chars = re.sub(pattern, '', text)
    
    return text_without_special_chars


def get_fundamentals(ticker):
    try:
        url = f"https://www.screener.in/company/{ticker}/consolidated/"
        request = requests.get(url)
        soup = BeautifulSoup(request.text, 'html.parser')
        fund_df, fund_changes = fund_analysis(soup, ticker)
    except:
        url = f"https://www.screener.in/company/{ticker}/"
        request = requests.get(url)
        soup = BeautifulSoup(request.text, 'html.parser')
        fund_df, fund_changes = fund_analysis(soup, ticker)
    return fund_df, fund_changes
    
def fund_analysis(soup, ticker):
    op_ = soup.findAll('div', class_='company-ratios')
    input_list = []
    for i in op_[0].find_all('span'):
        try:
            #print('----------------------')
            val = i.string.strip().replace(',','')
            input_list.append(val)
        except:continue
    #input_list = [inte if inte.isalpha() else int(float(inte)) for inte in input_list]
    input_list = input_list[:-1]
    my_list = []
    for i, val in enumerate(input_list):
        try:
            val = round(float(val))
            if type(val) == int and type(round(float(input_list[i+1]))) == int:
                val = round((val + round(float(input_list[i+1])))/2)
                input_list.pop(i+1)         
        except:
            val = val
        my_list.append(val)
    
    my_dict = {my_list[i]: my_list[i + 1] for i in range(0, len(my_list), 2)} 
    my_dict = {key:val for key,val in my_dict.items() if key not in ['Market Cap', 'High / Low']}
    fund_cols = my_dict.keys()
    fund_row = my_dict.values()
    fund_df = pd.DataFrame(fund_row).T
    fund_df.columns = fund_cols
    fund_df.index.names = [ticker]
    fund_df.rename(index={0: 'Numbers'}, inplace=True)
    
    dfs=[]
    op_growth = soup.findAll('table', class_='ranges-table')
    for op_g in op_growth:
        my_list_ = op_g.find_all(string=is_the_only_string_within_a_tag)
        my_list = my_list_[1:]
        my_list[-2] = 'Last Year'
        val = {my_list[i].replace(':',''): my_list[i + 1] for i in range(0, len(my_list), 2)}
        df = pd.DataFrame(val.items())
        df.set_index(df.columns[0], inplace=True) 
        df.index.name = None
        df.columns = [my_list_[0]]
        dfs.append(df)
        
    fund_changes = pd.concat(dfs, axis=1)
    return fund_df, fund_changes

def is_the_only_string_within_a_tag(s):
    """Return True if this string is the only child of its parent tag."""
    return (s == s.parent.string)

def display_market():
    url = f"https://www.moneycontrol.com/"
    request = requests.get(url)
    dic = {}
    soup = BeautifulSoup(request.text, 'html.parser')
    op_nif = soup.findAll('div', class_='MR5 clearfix')
    elements = op_nif[0].find_all(string=is_the_only_string_within_a_tag)
    market_data =  [[elements[i].upper(),float(elements[i+1]), float(elements[i+3])] for i in range(0, len(elements), 4)]
    return market_data 

def stock_data(tickers, period, interval):
    df = yf.download(tickers, period=period, interval=interval, rounding=True, progress=False).reset_index()
    try:
        df['Date'] = df['Date'].apply(lambda x: x.date())
    except:
        df['Date'] = df['Datetime'].apply(lambda x: x.date())
    df = df[['Date','Adj Close','Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date')
    df.columns = ['Adj Close', 'Open','High','Low','Close','Volume']
    return df

def plot_candle(name):
    period = st.select_slider("Select Period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'], value='6mo')
    interval = st.select_slider("Select Interval", options=["1h","1d","5d","1wk","1mo","3mo"], value='1d')
    df = stock_data(name+'.NS',period,interval)
    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.03, subplot_titles=(name, 'Volume'), 
                   row_width=[0.2, 0.7])

    # Plot OHLC on 1st row
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name=name), 
                    row=1, col=1
    )

    # Bar trace for volumes on 2nd row without legend
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False), row=2, col=1)

    # Do not show OHLC's rangeslider plot 
    fig.update(layout_xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# Gauge chart setup
def plot_gauge(test, df):
    latest_rsi = float(np.mean(df['RSI']))
    plot_bgcolor = "#def"
    quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"] 
    quadrant_text = ["", "<b>Extreme Overbought</b>", "<b>Momentum Zone</b>", "<b>Neutral</b>", "<b>Over Sold</b>", "<b>Extremely Oversold</b>"]
    n_quadrants = len(quadrant_colors) - 1

    min_value = 0
    max_value = 100  # RSI ranges from 0 to 100
    hand_length = np.sqrt(2) / 4
    hand_angle =  360 * (-latest_rsi/2 - min_value) / (max_value - min_value) - 180

    # Create gauge chart
    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",
                sort=False
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0,t=10,l=10,r=10),
            width=700,
            height=700,
            paper_bgcolor=plot_bgcolor,
            annotations=[
                go.layout.Annotation(
                    text=f"<b>{test} RSI Level:</b><br>{latest_rsi:.2f}",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.25, yanchor="bottom", yref="paper",
                    showarrow=False,
                    font=dict(size=14)
                )
            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(np.radians(hand_angle)),
                    y0=0.5, y1=0.5 + hand_length * np.sin(np.radians(hand_angle)),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )
    #st.plotly_chart(fig)
    html_code = fig.to_html(full_html=False, include_plotlyjs="cdn")
    st.components.v1.html(html_code, height=550)

def extract_date_and_time(datetime_str):
    # Regex pattern to match date (YYYY-MM-DD) and time (HH:MM:SS)
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", datetime_str)
    if match:
        date = match.group(1)  # Extract the date part
        time = match.group(2)  # Extract the time part
        return date, time
    return None, None

def extract_date_time_from_filename(filename):
    # Regex pattern to match the date and time
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}_\d{2}_\d{2})", filename)
    if match:
        date_str = match.group(1)  # Extract date
        time_str = match.group(2).replace("_", ":")  # Extract time and replace underscores with colons
        return date_str, time_str
    return None, None

def filter_list(my_lst):
    # Process the list
    filtered_data = []
    i = 0

    while i < len(my_lst):
        if isinstance(my_lst[i], str):
            if i + 1 < len(my_lst) and isinstance(my_lst[i + 1], int):
                filtered_data.extend([my_lst[i], my_lst[i + 1]])
            i += 2  # Skip both string and integer
        else:
            i += 1  # Skip invalid entries
    return filtered_data
    
def account_details():
    account_url = "https://tradetatics.blob.core.windows.net"
    account_key = "ZkAJWrsEc1GVv7n2QzZvGsEx5B2hcBcy4Nk+QjNvvAtb+ntX/mCB1zzDdxlDLc1Bhy8vySPNqcjd+AStgvf4OQ=="
    container_name = "daily-tech-analysis"
    return account_url, account_key, container_name

def get_current_date_time():
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    return current_date, current_time

def load_data():
    account_url, account_key, container_name = account_details()
    current_date, current_time = get_current_date_time()
    blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
    container_client = blob_service_client.get_container_client(container_name)
    try:
        file_name = f"Trend_Report_{current_date.replace('-','_')}.csv"
        blob_client = container_client.get_blob_client(file_name)
        downloaded_data = blob_client.download_blob().readall().decode("utf-8")
        stk_df = pd.read_csv(StringIO(downloaded_data))
    except:
        file_name = [blob.name for blob in container_client.list_blobs() if blob.name.endswith('csv')][-1]
        blob_client = container_client.get_blob_client(file_name)
        downloaded_data = blob_client.download_blob().readall().decode("utf-8")
        stk_df = pd.read_csv(StringIO(downloaded_data))
        
    forecast_json_path = 'daily_forecast.json'
    blob_client = container_client.get_blob_client(forecast_json_path)
    downloaded_data = blob_client.download_blob().readall().decode("utf-8")
    json_file = json.load(StringIO(downloaded_data))

    modified_time = blob_client.get_blob_properties().last_modified.astimezone(ist).strftime("%Y-%m-%d %H:%M:%S %Z")
    return modified_time, stk_df, json_file