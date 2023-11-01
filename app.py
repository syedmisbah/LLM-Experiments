from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatGooglePalm  # this will be used instead of ChatOpenAi above
import streamlit as st
import pandas as pd
# import os

# file_formats = {
#     "csv": pd.read_csv,
#     "xls": pd.read_excel,
#     "xlsx": pd.read_excel,
#     "xlsm": pd.read_excel,
#     "xlsb": pd.read_excel,
# }


# def clear_submit():
#     """
#     Clear the Submit Button State
#     Returns:

#     """
#     st.session_state["submit"] = False


# @st.cache_data(ttl="2h")
# def load_data(uploaded_file):
#     try:
#         ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
#     except:
#         ext = uploaded_file.split(".")[-1]
#     if ext in file_formats:
#         return file_formats[ext](uploaded_file)
#     else:
#         st.error(f"Unsupported file format: {ext}")
#         return None


st.set_page_config(
    page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜", layout="wide"
)
st.title("🦜 LangChain: Chat with pandas DataFrame")

# uploaded_file = st.file_uploader(
#     "Upload a Data file",
#     type=list(file_formats.keys()),
#     help="Various File formats are Support",
#     on_change=clear_submit,
# )

# if not uploaded_file:
#     st.warning(
#         "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
#     )

# if uploaded_file:
#     df = load_data(uploaded_file)

# print(os.getcwd())
df = pd.read_csv("data/train.csv")  # TODO add more here if you want

openai_api_key = "REMOVED"  ##TODO:commented once Google chat LLM is used
google_api_key = "Add Google Key Here"  # TODO Add google key here


if "messages" not in st.session_state or st.sidebar.button(
    "Clear conversation history"
):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    llm_openai = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        streaming=True,
    )

    llm_google = ChatGooglePalm(
        temperature=0,
        model="chat-bison-001",
        google_api_key=google_api_key,
        streaming=True,
    )  # TODO Probably remove the dash from bison-001

    pandas_df_agent = create_pandas_dataframe_agent(
        llm_openai,  # TODO change this to llm_google
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
