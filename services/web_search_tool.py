import asyncio
import json
from typing import Optional, Any
import unicodedata

from langchain_community.tools.google_serper import GoogleSerperResults

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI


from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


from langchain_core.callbacks import CallbackManagerForToolRun
import streamlit as st

from services.scraping import scrape_text_async
from lib.utils import filecontent

SUMMARY_TEMPLATE = filecontent("ChatPromptTemplate.txt")
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


class WebSearchTool(GoogleSerperResults):
    description: str = (
        "A low-cost Google Search API."
        "Useful for when you need to answer questions about current events."
        "Input should be a list of search queries seperated by a '|' character."
        "Output is a JSON object of with a list of results it found by scraping"
    )

    def __init__(self, max_results: int = 10):
        super().__init__()
        self.api_wrapper.k = max_results

    @staticmethod
    def decode_query(query):
        decoded_query = query.encode().decode('unicode_escape')
        normalized_query = unicodedata.normalize('NFD', decoded_query)
        ascii_folded_query = ''.join(c for c in normalized_query if unicodedata.category(c) != 'Mn')

        return ascii_folded_query

    @staticmethod
    async def summarize(scrapes: tuple[Any], questions):
        summary_chain = RunnablePassthrough.assign(
            text=lambda x: x["t"],
            question=lambda x: x["q"]
        ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-4-turbo") | StrOutputParser()
        main_chain = RunnablePassthrough() | summary_chain.map()
        return await main_chain.ainvoke([{"t": t, "q": q} for (t, q) in zip(scrapes, questions)])

    async def search_async(self, query: str):
        spinner = st.spinner("Browsing on web: " + query)
        spinner.__enter__()

        queries = query.split("|")
        query_searches = [self.api_wrapper.aresults(q) for q in queries]

        values = await asyncio.gather(*query_searches)

        organics = [(v["organic"], query) for (v, query) in zip(values, queries)]
        links = list(set([(r["link"], q) for (o, q) in organics for r in o]))
        links = [(link, q) for (link, q) in links if ".pdf" not in link]
        scrapes = [scrape_text_async(link) for (link, q) in links]
        scraped = await asyncio.gather(*scrapes)
        scraped = await self.summarize(scraped, [q for (link, q) in links])

        spinner.__exit__(None, None, None)

        return json.dumps({
            "results": [{"query": l[1], "text": t[:10000]} for (l, t) in zip(links, scraped)]
        })

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(self.search_async(query))
