import faiss
import pymupdf
import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils._logger import logger
from .utils._validation import config_args

documents = []

SYSTEM_PROMPT = (
    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. If you don"t know the answer, say that you don"t know.
Keep the answer concise.\n\n"""
    "{context}"
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)

st.set_page_config("PDF Context")

# Helper Functions


def create_vector_store():
    """Initialize vector store."""
    try:
        return FAISS(
            embedding_function=st.session_state.embeddings,
            index=faiss.IndexFlatL2(
                len(st.session_state.embeddings.embed_query("Hello World"))
            ),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")


def create_text_splitter():
    """Creates and configures the text splitter."""
    try:
        return RecursiveCharacterTextSplitter(
            chunk_size=config_args.chunk_size, chunk_overlap=config_args.chunk_overlap
        )
    except Exception as e:
        logger.error(f"Error creating text splitter: {e}")
        return None


def add_documents_to_vectorstore(document: list[str]):
    """Adds documents to the FAISS vectorstore."""
    try:
        st.session_state.vector_store.add_texts(texts=document)
    except Exception as e:
        logger.error(f"Error adding documents to vectorstore: {e}")


def create_question_chain():
    """Creates a chain for passing documents to model."""
    try:
        return create_stuff_documents_chain(st.session_state.model, PROMPT_TEMPLATE)
    except Exception as e:
        logger.error(f"Error creating document chain: {e}")


def create_retrieval():
    """Creates the retrieval chain."""
    try:
        return create_retrieval_chain(
            st.session_state.retriver, create_question_chain()
        )
    except Exception as e:
        logger.error(f"Error creating retrieval chain: {e}")
        return None


def process_uploaded_pdf(pdf_bytes: bytes):
    """Extracts text from a PDF and splits it into chunks."""
    try:
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in pdf:
            text += "\n\n" + page.get_text()
        pdf.close()
        text_splitter = create_text_splitter()
        if text_splitter is None:
            logger.error("Text splitter is None. Cannot proceed.")
            return []
        chunks = text_splitter.split_text(text)
        add_documents_to_vectorstore(chunks)
    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {e}")
        return []


def respond(query):
    """Retrieves context based on the query and returns the answer."""
    try:
        retriver_chain = create_retrieval()
        if retriver_chain is None:
            logger.error("Retrieval chain is None. Cannot proceed.")
            return ""
        for chunk in retriver_chain.stream({"input": query}):
            if chunk.get("context"):
                documents.append(chunk)
            elif chunk.get("answer"):
                yield chunk["answer"]
    except Exception as e:
        logger.error(f"Error responding to query: {e}")
        return ""


# Main Application Logic
if "model" not in st.session_state:
    st.session_state["model"] = ChatOllama(
        model=config_args.base_model, base_url=config_args.local_url
    )
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = OllamaEmbeddings(
        model=config_args.embedding_model, base_url=config_args.local_url
    )
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = create_vector_store()
if "retriver" not in st.session_state:
    st.session_state["retriver"] = st.session_state.vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = False

# File Uploader
with st.sidebar:
    with st.spinner():
        uploaded_pdf = st.file_uploader("Upload your document (pdf)", type="pdf")
        if uploaded_pdf and not st.session_state["uploaded"]:
            try:
                pdf_bytes = uploaded_pdf.read()
                process_uploaded_pdf(pdf_bytes)
                st.session_state["uploaded"] = True
            except Exception as e:
                logger.error(f"Error processing uploaded PDF: {e}")

# Interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Type here..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(respond(prompt))
        st.session_state["messages"].append({"role": "assistant", "content": response})

        st.write({"Resource": documents[0]["context"][0].page_content})
        documents = []
