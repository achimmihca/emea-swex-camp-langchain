import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
    prompt_template = PromptTemplate(input_variables=input_variables, template=template)

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

# Streamlit app definition
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Board Game Q&A Wizard")
    st.write("This app allows you to ask questions about the content from the documents loaded from the given PDFs.")

    # File uploader for PDF files
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Save the uploaded files to disk
        pdf_paths = []
        for uploaded_file in uploaded_files:
            target_path = "uploaded-files/" + uploaded_file.name
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(target_path)

        # Load and process the PDF documents
        retriever = load_and_process_documents(pdf_paths)

        # Create the RAG chain for question answering
        rag_chain = create_rag_chain(retriever)
    
        # Input box for user to enter their question
        question = st.text_input("Enter your question:") 
    
        if question:
            # Show a spinner while retrieving and generating the answer
            with st.spinner("Retrieving and generating answer..."):
                # Invoke the RAG chain with the user's question
                result = rag_chain.invoke(question)
            # Display the answer
            st.write("**Answer:**")
            st.write(result)

# Run the Streamlit app
if __name__ == "__main__":
    main()