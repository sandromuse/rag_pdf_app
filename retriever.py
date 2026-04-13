from dotenv import load_dotenv #import a fucntion that helps me read the env file to use my key 
from langchain_community.embeddings import HuggingFaceEmbeddings # imports a functions that embedds each chunk as a number
from langchain_community.vectorstores import Chroma # a vector database that store the embeddings the chunks 

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2") 
vectorstore =  Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def retrieve(question):
        return vectorstore.similarity_search(question,k = 3)