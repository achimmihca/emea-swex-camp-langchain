import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

# Set up environment variables for API keys and endpoints
load_dotenv()

# Load and process documents function


@st.cache_resource
def load_and_process_documents(pdf_path: str):
    """
    Load and process the PDF document into retrievable chunks using vector embeddings.
    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        retriever: A retriever object for querying the document.
    """
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split the text into chunks with overlap for context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings for the document chunks and store them in a vector store
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings())

    # Return the retriever object for querying
    return vectorstore.as_retriever()

# Function to create the RAG (Retrieval-Augmented Generation) chain


def create_rag_chain(retriever):
    """
    Create a RAG chain for question answering using retrieved document chunks and LLM.
    Args:
        retriever: A retriever object for querying the document.

    Returns:
        rag_chain: A chain object that can process questions and generate answers.
    """
    # Load the prompt template for the RAG chain
    # prompt = hub.pull("rlm/rag-prompt")

    # Define the input variables
    input_variables = ['context', 'question']

    # Define the template
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:"""

    # Create the PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=input_variables, template=template)

    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Function to format the retrieved documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


# Streamlit app definition


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("RAG Chain Question Answering")
    st.write("This app allows you to ask questions about the content from the document loaded from the given PDF.")

    # Specify the path to the PDF file
    pdf_path = "dnd.pdf"
    # Load and process the PDF document
    retriever = load_and_process_documents(pdf_path)

   # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_perspectives
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))


    )

    # Retrieve
    # Input box for user to enter their question
    question = st.text_input("Enter your question:")
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question": question})
    len(docs)

    # Create the RAG chain for question answering
    # RAG
    template = """Answer the following question based on this context:

{context}

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
        {"context": retrieval_chain,
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    if question:
        # Show a spinner while retrieving and generating the answer
        with st.spinner("Retrieving and generating answer..."):
            # Invoke the RAG chain with the user's question
            result = final_rag_chain.invoke({"question": question})
        # Display the answer
        st.write("**Answer:**")
        st.write(result)


# Run the Streamlit app
if __name__ == "__main__":
    main()
