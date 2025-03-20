import os
# from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
# from langchain import hub
from langgraph.graph import START, StateGraph
from pinecone import Pinecone
from typing_extensions import List, TypedDict
import edgar
import streamlit as st

# load_dotenv()

# -------------------------------------------------------------------
os.environ["LANGSMITH_TRACING"] = st.secrets['LANGSMITH_TRACING']
os.environ["LANGSMITH_API_KEY"] = st.secrets['LANGSMITH_API_KEY']

# edgar.set_identity(os.getenv('MY_EMAIL'))
edgar.set_identity(st.secrets['MY_EMAIL'])


def get_latest_10k(ticker):
    company = edgar.Company(ticker)
    ten_k = company.latest('10-K').obj()

    filing_date = ten_k.filing_date.strftime('%Y-%m-%d')
    form = ten_k.form

    wanted_items = ['Item 1', 'Item 1A', 'Item 7']
    docs = [
        Document(
            page_content=ten_k[item],
            metadata={
                "ticker": ticker,
                "form": form,
                "filing_date": filing_date,
                "section": item,
            }
        )
        for item in wanted_items
    ]

    return docs


# -------------------------------------------------------------------
# Initialize LLM models:
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")

# Data ----------------------------------------------------------------
# Initialize the vector store:
# pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
pc = Pinecone(api_key=st.secrets['PINECONE_API_KEY'])
index_name = 'tenkbot'
index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# Get the original text data:
add_docs = False
if add_docs:
    docs = get_latest_10k('TSLA')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True)
    split_docs = text_splitter.split_documents(docs)

    split_doc_ids = [
        '|'.join(str(val) for val in doc.metadata.values())
        for doc in split_docs
    ]

# Agent -----------------------------------------------------------------
# rag_prompt = hub.pull('rlm/rag-prompt')
rag_prompt = ChatPromptTemplate([
    'human',
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:
"""
])


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.invoke(
        {"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)

    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Streamlit ---------------------------------------------------------
st.title('10-K bot')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

resp = None
# React to user input
user_prompt = st.chat_input("Ask questions about TSLA.")
if user_prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    resp = graph.invoke({'question': user_prompt})

if resp:
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(resp['answer'])

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": resp['answer']})
