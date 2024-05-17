from argparse import Namespace
from typing import Type, List

import streamlit as st
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

from langchain_core.tools import BaseTool
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from services.stream_handler import StreamHandler



@tool
def no_op_tool(input: str) -> str:
    """I don't do anything."""
    return "Nothing"


class AgentSystem:
    """Agent System to handle the user input and output."""
    stream_handler: StreamHandler

    def __init__(self):
        self.main_agent = None
        self.summary_prompt = None
        self.input_label = ""
        self.namespace = ""

    @staticmethod
    def _get_container():
        print("CONTAINER WAS GRANTED")
        return st.empty()

    def init(self):
        """Initialize the stream handler."""
        self.stream_handler = StreamHandler(self._get_container)

    def __user_input(self):
        self.init()

        for user_msg, ai_msg in st.session_state[self.namespace].history:
            st.chat_message("user").write(user_msg)
            st.chat_message("assistant").write(ai_msg)

        if user_input := st.chat_input(self.input_label):
            st.chat_message("user").write(user_input)

            self.main_agent.invoke(
                {"input": user_input},
                config={"callbacks": [self.stream_handler]}
            )

            st.session_state[self.namespace].history.append((user_input, self.stream_handler.text))

    def loop(self):
        """Start the agent loop."""
        self.__user_input()


class AgentSystemBuilder:
    """Builder class to create an Agent System."""
    def __init__(self, namespace: str, agent: Type[AgentSystem], clear: bool):
        self.namespace = namespace
        self.summary_prompt = None
        self.prompt = None
        self.tools = [no_op_tool]
        self.agent_system = agent()
        self.chat_llm = Type[BaseChatModel]
        self.model = "gpt-4-turbo"
        if clear:
            st.session_state[self.namespace] = Namespace()

    def with_chat_llm(self, chat_llm: Type[BaseChatModel]):
        """Set the chat language model"""
        self.chat_llm = chat_llm
        st.session_state[self.namespace].chat_llm = self.chat_llm
        return self

    def with_system_template(self, template_file: str = None, template_string: str = None, variables: dict = {}):
        """Set the system template for the agent."""
        if template_file:
            system_prompt_template = SystemMessagePromptTemplate.from_template_file(
                template_file=template_file, input_variables=list(variables.keys() if variables else [])
            )
        else:
            system_prompt_template = SystemMessagePromptTemplate.from_template(template=template_string)
            st.session_state[self.namespace].system_prompt = system_prompt_template.format().content

        self.prompt = (
                system_prompt_template.format(**variables)
                + MessagesPlaceholder("chat_history", optional=True)
                + HumanMessagePromptTemplate.from_template("{input}")
                + MessagesPlaceholder("agent_scratchpad")
        )

        st.session_state[self.namespace].variables = variables

        return self

    def with_summary_template(self, summary_template: str = None, summary_text: str = None, input:  str = None):
        """Set the summary template for the agent."""
        if summary_text:
            self.summary_prompt = (
                HumanMessagePromptTemplate.from_template(summary_text)
                .format(input=input)
                .content
            )
        else:
            self.summary_prompt = (
                HumanMessagePromptTemplate.from_template_file(
                    template_file=summary_template, input_variables=[]
                )
                .format()
                .content
            )

        return self

    def with_model(self, model: str):
        """Set the model for the agent."""
        self.model = model

        return self

    def with_tools(self, tools: List[BaseTool]):
        """Set the tools for the agent."""
        self.tools = tools
        return self

    def build(self):
        """Build the agent system."""
        if "history" not in st.session_state[self.namespace]:
            st.session_state[self.namespace].history = []

        llm = self.chat_llm(model=self.model, temperature=0)

        agent = create_openai_tools_agent(llm, self.tools, self.prompt)

        if "memory" not in st.session_state[self.namespace]:
            st.session_state[self.namespace].memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        memory = st.session_state[self.namespace].memory

        # Create an agent executor by passing in the agent and tools
        st.session_state[self.namespace].agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=False, memory=memory
        )

        self.agent_system.main_agent = st.session_state[self.namespace].agent_executor
        self.agent_system.summary_prompt = self.summary_prompt
        self.agent_system.namespace = self.namespace
        return self.agent_system
