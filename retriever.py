from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def retrieve(question):
    vectorstore = Chroma(embedding_function=embeddings)
    return vectorstore.similarity_search(question, k=3)