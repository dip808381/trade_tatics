import streamlit as st
# from home import *
# from filter_df import *
# from chat import *

#streamlit page
home = st.Page("home.py", title='Home',icon=":material/home:")
quick_filter = st.Page("quick_filter.py", title='QUICK FILTER',icon="📃")
filter_df = st.Page("filter_df.py", title='SCREENER',icon="📈")
chat = st.Page("chat.py", title='CHATBOT',icon="💬")


# Navigation
pg = st.navigation([home, quick_filter, filter_df, chat])

pg.run()