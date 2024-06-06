""" Company Analyzer Page, combining the CompanyAnalyzerAgent and the CompanyAnalyzerAgentSystemBuilder """

from argparse import Namespace

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from services.agent_system import AgentSystemBuilder, AgentSystem
#from services.bad_tool import even_odd, fahrenheit_to_cel

from services.web_search_tool import WebSearchTool

from lib.utils import assetpath


import lib.environment as keys

keys.initialise()

# Page Header
st.header("Chat with Llama3")

NAMESPACE = "swex"

if not st.session_state.get(NAMESPACE):
    st.session_state[NAMESPACE] = Namespace()

agent_already_existed = "agent_system" in st.session_state[NAMESPACE]

if not agent_already_existed:
    SYSTEM_PROMPT_PATH = assetpath("system-prompt.txt")

    st.session_state[NAMESPACE].agent_system = (
        AgentSystemBuilder(namespace=NAMESPACE, agent=AgentSystem, clear=not agent_already_existed)
        .with_system_template(SYSTEM_PROMPT_PATH)
        .with_chat_llm(ChatOllama)
        .with_model("llama3")
        .with_temperature(0)
        #.with_tools([WebSearchTool(max_results=2), even_odd, fahrenheit_to_cel])
        .build()

    )

st.session_state[NAMESPACE].agent_system.loop()
