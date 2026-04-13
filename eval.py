from chain import get_answer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pypdf import PdfReader
import json
load_dotenv()


def generate_test_cases():
    reader = PdfReader("hamster.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    Prompt =  f"""Given this text: {text} Generate 5 question and answer pairs.Return ONLY a JSON array in this exact format, no other text:[{{"question": "...", "expected": "..."}}]"""
    res = llm.invoke(Prompt)
    return json.loads(res.content)

test_cases = generate_test_cases()

def judge_answer(question,expected,actual):
    judge = ChatGroq(model_name="llama-3.3-70b-versatile")
    comparison_prompt =  f"take the actual answer:{actual} compare it to expcted:{expected}. If they are close to each other return Yes else No"
    res = judge.invoke(comparison_prompt)
    return 'Yes' in res.content

eval_score = 0
for case in test_cases:
     actual = get_answer(case["question"])
     if judge_answer(case["question"],case["expected"],actual):
        eval_score += 1

print(f"{eval_score}/{len(test_cases)}")