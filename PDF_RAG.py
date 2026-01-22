from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st


st.set_page_config(
    page_title="PDF RAG Chatbot",
    layout="wide"
)

st.title("PDF Question Answering.")
st.caption("Local RAG system using LangChain, FAISS, and Ollama.")

st.sidebar.header("Settings")

retrieval_type = st.sidebar.selectbox(
    "Retrieval Strategy",
    ["mmr", "similarity"]
)

k_docs = st.sidebar.slider("Top-k Retrieved Chunks", 3, 10, 5)

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

def format_docs (retrieved_docs):
    context_blocks = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "N/A")
        text = doc.page_content
        context_blocks.append(f"[Page: {page}] {text}")
    return "\n\n".join(context_blocks)

def format_history(history):
    if not history:
        return "No previous conversation"
    formatted=[]
    for msg in history[-6:]: #keeps last 6 messages
         role = msg["role"]
         content = msg["content"]
         formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted) 

def load_history(_):
    if not hasattr (st.session_state, "chat_history"):
        return "No previous conversation"
    return format_history(st.session_state.chat_history)


@st.cache_resource(show_spinner=True)
def build_rag_chain(file_path, retrieval_type, k_docs, _session_state):

    #pdf loader to get text
    loader = PyMuPDFLoader(file_path)

    #loads text in docs as document object
    docs = loader.load()

    #load the splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    #load embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
    )

    #create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    #create a retriever
    if retrieval_type == "mmr":
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k':k_docs, "fetch_k":20, "lambda_mult":0.7 })
    else:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k':k_docs})

    #load the llm
    llm = ChatOllama(
        model="llama3.1:8b"                    
    )

    #prompt
    prompt = PromptTemplate(
        template="""
            You are an expert tutor helping the student understand PDF content.

            Instructions:
            - Answer ONLY using the context.
            - Explain clearly and step-by-step when helpful.
            - Cite the page number where applicable (eg., "According to page 5...")
            - If the context doesn't contain enough information, say "I don't have enough information in the PDF to answer that."
            - Be concise but thorough

            Conversation History:
            {history}
            
            Context:
            {context}

            Question:
            {question}
            Answer: """,
        input_variables=['history', 'context', 'question']
    )

    #parallel chain
    parallel_chain = RunnableParallel({
        'context' : retriever | RunnableLambda(format_docs),
        'history' : RunnableLambda(load_history),
        'question' : RunnablePassthrough()
    })

    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser

    return chain

if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_path = os.path.join("temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.rag_chain = build_rag_chain(pdf_path, retrieval_type, k_docs, st.session_state)
    st.success("PDF processed successfully! Ask your question below.")

#display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#chat interface
if st.session_state.rag_chain:
    user_query = st.chat_input("Ask a question about the PDF")

    if user_query:
        st.session_state.messages.append({"role":"user", "content":user_query})
        st.chat_message("user").write(user_query)

        with st.spinner("Generating answer..."):
            answer = st.session_state.rag_chain.invoke(user_query)

        st.session_state.messages.append({"role":"assistant", "content":answer})   
        st.chat_message("assistant").write(answer)

        st.session_state.chat_history.append(
            {"role":"user", "content":user_query}
        )
        st.session_state.chat_history.append({"role":"assistant", "content":answer})
else:
    st.info("Upload a PDF from sidebar to begin.")
