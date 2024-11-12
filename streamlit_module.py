import re
import requests
import pandas as pd
from io import StringIO
import streamlit as st
from datetime import datetime
from bs4 import BeautifulSoup
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from azure.storage.blob import BlobServiceClient
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
    # data.rename(columns=rename_columns(), inplace=True)
    ser_name = st.selectbox('GET ALL TECHNICAL AND FUNDAMENTAL ANALYSIS..', ['COMPANY_NAME','TICKER'], index=None, placeholder='search by company or ticker..')

    if ser_name == 'COMPANY_NAME':
        stk_comp = st.selectbox('Search by company name..', list(data['COMPANY_NAME'].unique()), index=None, placeholder='search by company..')
        if stk_comp:
            df_comp = data[data['COMPANY_NAME'].isin([stk_comp])]
            df_comp = df_comp.rename(columns=col_name_change)
            df_comp = df_comp.T
            df_comp.reset_index(inplace=True)
            df_comp.columns = ['COLUMNS', 'VALUES']
            stk_tic = df_comp.loc[df_comp['COLUMNS'] == 'TICKER']
            stk_tic_val = stk_tic['VALUES'][0]
            plot_candle(stk_tic_val)
            col1, col2 = st.columns(2)
            with col1:
                st.table(df_comp[:len(df_comp)//2])
            with col2:
                st.table(df_comp[len(df_comp)//2:])

            fund_ana = st.toggle('Check fundamental analysis..')
            if fund_ana:
                #stk_tic = df_comp.loc[df_comp['COLUMNS'] == 'STOCK']
                df_fund = pd.DataFrame(fund_prft(stk_tic_val).items())
                df_fund.columns = ['COLUMNS', 'VALUES']
                fund_col1, fund_col2 = st.columns(2)
                with fund_col1:
                    st.table(df_fund[:len(df_fund)//2])
                with fund_col2:
                    st.table(df_fund[len(df_fund)//2:])

    elif ser_name == 'TICKER':
        stk_tic = st.selectbox('Search by ticker..', list(data['STOCK'].unique()), index=None, placeholder='search by ticker..')
        if stk_tic:
            df_tic = data[data['STOCK'].isin([stk_tic])]
            df_tic = df_tic.rename(columns=col_name_change)
            df_tic = df_tic.T
            df_tic.reset_index(inplace=True)
            df_tic.columns = ['COLUMNS', 'VALUES']
            plot_candle(stk_tic)
            col3, col4 = st.columns(2)
            st.write(len(df_tic))
            with col3:
                st.table(df_tic[:len(df_tic)//2])
            with col4:
                st.table(df_tic[len(df_tic)//2:])

            fund_ana = st.toggle('Check fundamental analysis..')
            if fund_ana:
                df_fund = pd.DataFrame(fund_prft(stk_tic).items())
                df_fund.columns = ['COLUMNS', 'VALUES']
                fund_col1, fund_col2 = st.columns(2)
                with fund_col1:
                    st.table(df_fund[:len(df_fund)//2])
                with fund_col2:
                    st.table(df_fund[len(df_fund)//2:])

    




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

def remove_special_characters(text):
    # Define the pattern for special characters using regular expression
    pattern = r'[^a-zA-Z0-9\s]'  # This pattern matches anything that is not alphanumeric or whitespace

    # Use re.sub() to replace the special characters with an empty string
    text_without_special_chars = re.sub(pattern, '', text)
    
    return text_without_special_chars

def fund_analysis(soup):
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
    return my_dict

def fund_prft(stock):
    for i in range(10):
        ticker = stock
        url = f"https://www.screener.in/company/{ticker}/"
        #url = f"https://www.screener.in/company/{ticker}/"
        request = requests.get(url)
        dic = {}
        soup = BeautifulSoup(request.text, 'html.parser')
        op = soup.findAll('tr', class_='strong')
        for i in op:
            try:
                key =i.td.string.strip()
            except:
                key = remove_special_characters(i.td.text.strip().replace('\xa0', ''))

            value = []
            for val in i.findAll('td', class_=['', "highlight-cell"]):
                try:
                    value.append(int(float(val.text.replace(',', ''))))
                except:continue
            #value = [int(float(val.text.replace(',', ''))) for val in i.findAll('td', class_='')]
            #percent = round((len([a for a in value if a%2==0]) / len(value)) * 100, 2)
            #print(key, value)
            percent = growth(value)

            if key not in dic:
                #key = key+'_qtr'
                dic[key] = percent
            else:
                key = key+'_yr'
                dic[key] = percent
        if len(dic) == 0:
            continue
        else:
            break
    dic_ = fund_analysis(soup)
    fin_dic = Merge(dic_, dic)
    return {key.upper():round(val, 2) for key,val in fin_dic.items() if key not in ['Market Cap', 'High / Low']}

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

def plot_candle(name, period='1y', interval='1d'):
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
        file_name = [blob.name for blob in container_client.list_blobs()][-1]
        blob_client = container_client.get_blob_client(file_name)
        downloaded_data = blob_client.download_blob().readall().decode("utf-8")
        stk_df = pd.read_csv(StringIO(downloaded_data))
        
    return file_name, stk_df