import streamlit as st
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits import create_csv_agent

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv('.env'))
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
load_dotenv()

tmp = 'tmp'
file_name = [file for file in os.listdir(tmp) if file.endswith('csv')][-1]

# app config
st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.markdown("<h1 style='text-align: center; font-size: 30px;'>Need Market Insights? Just Ask!</h1>", unsafe_allow_html=True)


# Styling to center radio options
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

def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI()
        
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query,
    })
    return response

def get_response_from_csv(user_query):
    csv_df = os.path.join(tmp, file_name)
    llm = ChatOpenAI(temperature=0.5,max_tokens=1000)
    agent_executer = create_csv_agent(llm, csv_df, allow_dangerous_code=True)
    response = agent_executer.run(user_query)
    return response

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

options = ['General', 'Learning', 'Technical', 'Fundamental']
chat_options = st.radio('Select options you want to explore today..', options,index=0, horizontal=True)

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if chat_options == 'General':
            response = get_response(user_query, st.session_state.chat_history)
            st.markdown(response)
        elif chat_options == 'Learning':
            response = get_response(user_query, st.session_state.chat_history)
            st.markdown(response)
        elif chat_options == 'Technical':
            response = get_response_from_csv(user_query)
            st.markdown(response)
        elif chat_options == 'Fundamental':
            response = get_response(user_query, st.session_state.chat_history)
            st.markdown(response)
        else:
            response = get_response(user_query, st.session_state.chat_history)
            st.markdown(response)

    st.session_state.chat_history.append(AIMessage(response))


   