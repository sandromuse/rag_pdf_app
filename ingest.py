import os #import os functions

from dotenv import load_dotenv #import a fucntion that helps me read the env file to use my key 
from langchain_community.document_loaders import PyPDFLoader #imports a function that loads my pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter # imports a function that splits my file into chunks
from langchain_community.embeddings import HuggingFaceEmbeddings # imports a functions that embedds each chunk as a number
from langchain_community.vectorstores import Chroma # a vector database that store the embeddings the chunks 

load_dotenv() # acces the key for use here
def ingest_document(pdf_path): 
    loader = PyPDFLoader(pdf_path) # acces the pdf
    documents = loader.load() # actuallt laods the pdf here
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50) # gets the spliiter which will turn file  into chunks oo the size 500 , wich overlapt with each other by 50
    chunks = splitter.split_documents(documents) # actually chunks it
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2") # get the emdedings model that run locally 
    vectorstore = Chroma.from_documents(documents = chunks, embedding = embeddings, persist_directory = './chroma_db' ) # resulting pdf file into chunks and its embedding stored in vectors in chromadb
    return vectorstore