from retriever import retrieve
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

def get_answer(question, chat_history=[]):
    docs = retrieve(question)
    context = " ".join([doc.page_content for doc in docs])
    history_text = ""
    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"
    prompt = f"Use the following context to answer the question. Context: {context} answer this question {question}. Here are the previous question {history_text}"
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    response = llm.invoke(prompt)
    return response.content