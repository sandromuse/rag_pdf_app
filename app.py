import streamlit as st
import tempfile
from chain import get_answer
from ingest import ingest_document

st.title("Chat with your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_vectorstore(path):
    return ingest_document(path)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    load_vectorstore(tmp_path)
    st.success("PDF uploaded and processed!")

for chat in st.session_state.messages:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

question = st.chat_input("Ask a question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    answer = get_answer(question, st.session_state.messages[:-1])
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)