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
def load_and_process_documents(pdf_paths: list):
    """
    Load and process the PDF documents into retrievable chunks using vector embeddings.
    Args:
        pdf_paths (list): A list of paths to the PDF files.

    Returns:
        retriever: A retriever object for querying the document.
    """
    all_splits = []

    for pdf_path in pdf_paths:
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split the text into chunks with overlap for context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        all_splits.extend(splits)

    # Create embeddings for the document chunks and store them in a vector store
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings())

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


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

# Streamlit app definition


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Board Game Q&A Wizard - Rank Fusion")
    st.write("This app allows you to ask questions about the content from the documents loaded from the given PDFs.")

    # File uploader for PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Save the uploaded files to disk
        pdf_paths = []
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(uploaded_file.name)

        # Load and process the PDF documents
        retriever = load_and_process_documents(pdf_paths)

        # RAG-Fusion: Related
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        # Retrieve
        # Input box for user to enter their question
        question = st.text_input("Enter your question:")
        retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
        docs = retrieval_chain_rag_fusion.invoke({"question": question})
        len(docs)

        # Create the RAG chain for question answering
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(temperature=0)
        final_rag_chain = (
            {"context": retrieval_chain_rag_fusion,
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
