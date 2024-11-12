import streamlit as st
import pandas as pd
import numpy as np
import shutil
from datetime import date, timedelta

# from home import *
# from filter_df import *
# from chat import *


#streamlit page
home = st.Page("home.py", title='Home',icon=":material/home:")
filter_df = st.Page("filter_df.py", title='SCREENER',icon="📈")
chat = st.Page("chat.py", title='CHAT',icon="💬")


# Navigation
pg = st.navigation([home, filter_df, chat])

pg.run()