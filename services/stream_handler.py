from langchain_core.callbacks import BaseCallbackHandler
import streamlit as st
from langchain_core.outputs import LLMResult


class StreamHandler(BaseCallbackHandler):
    def __init__(self, get_container_func, initial_text=""):
        self.get_container = get_container_func
        self.text = initial_text
        self.container = None
        self.spinner = None

    def on_chat_model_start(self, *args, **kwargs) -> None:
        if not self.container:
            self.container = self.get_container()
        self.spinner = st.spinner("Gathering results...")
        self.spinner.__enter__()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.spinner.__exit__(None, None, None)
        self.text += token
        if not self.text.strip() == "":
            self.container.chat_message("assistant").write(self.text)

    def on_llm_end(self, response, **kwargs) -> LLMResult:
        print("End  was called")
        return response
