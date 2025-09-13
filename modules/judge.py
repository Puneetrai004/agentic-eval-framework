from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="google/flan-t5-small")

def llm_judge(prompt, response):
    llm = load_llm()
    eval_prompt = f"Evaluate the following.\nPrompt: {prompt}\nResponse: {response}\nGive a score (0-1) with reasoning."
    output = llm(eval_prompt, max_new_tokens=50)
    return output[0]["generated_text"]
