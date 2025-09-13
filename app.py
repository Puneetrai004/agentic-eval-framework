import os
import streamlit as st
import pandas as pd

from modules.dataset import generate_synthetic_data
from modules.scoring import (
    score_instruction_following, score_hallucination,
    score_assumption_control, score_coherence_accuracy
)
from modules.visualization import show_leaderboard
from modules import judge

# Force Streamlit to use /app/.streamlit instead of /.streamlit
os.environ["STREAMLIT_HOME"] = os.path.join(os.getcwd(), ".streamlit")
os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(os.getcwd(), ".streamlit")
os.environ["STREAMLIT_USER_HOME"] = os.path.join(os.getcwd(), ".streamlit")

# Make sure directory exists
os.makedirs(os.environ["STREAMLIT_HOME"], exist_ok=True)

st.set_page_config(page_title="Agentic Evaluation Framework", layout="wide")
st.title("🤖 Agentic Evaluation Framework")

tab1, tab2, tab3 = st.tabs(["Evaluation", "LLM-as-Judge (Optional)", "Dataset"])

with tab1:
    st.header("Automatic Scoring")
    df = generate_synthetic_data(100)
    scores = []
    for _, row in df.iterrows():
        s1, e1 = score_instruction_following(row["prompt"], row["response"])
        s2, e2 = score_hallucination(row["response"], ["Here is a correct and concise answer."])
        s3, e3 = score_assumption_control(row["response"])
        s4, e4 = score_coherence_accuracy(row["response"])

        scores.extend([
            {"agent_id": row["agent_id"], "metric": "Instruction", "score": s1, "explanation": e1},
            {"agent_id": row["agent_id"], "metric": "Hallucination", "score": s2, "explanation": e2},
            {"agent_id": row["agent_id"], "metric": "Assumption", "score": s3, "explanation": e3},
            {"agent_id": row["agent_id"], "metric": "Coherence", "score": s4, "explanation": e4},
        ])

    df_scores = pd.DataFrame(scores)
    st.dataframe(df_scores)
    show_leaderboard(df_scores)

with tab2:
    st.header("Optional: LLM-as-Judge")
    st.info("⚠️ Uses flan-t5-small — may be slow on free Spaces.")
    if st.checkbox("Enable LLM Judge?"):
        for _, row in df.head(5).iterrows():  # limit for speed
            llm_eval = judge.llm_judge(row["prompt"], row["response"])
            st.write(f"Prompt: {row['prompt']} | Response: {row['response']} | Judge says: {llm_eval}")

with tab3:
    st.header("Synthetic Dataset")
    st.dataframe(df)
