import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def show_leaderboard(df_scores):
    st.subheader("Leaderboard (Average Scores)")
    leaderboard = df_scores.groupby("agent_id")["score"].mean().sort_values(ascending=False).reset_index()
    st.dataframe(leaderboard)

    st.subheader("Heatmap of Agent Performance")
    pivot = df_scores.pivot_table(index="agent_id", columns="metric", values="score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(pivot, annot=True, cmap="viridis", ax=ax)
    st.pyplot(fig)

    st.subheader("Score Distribution by Metric")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x="metric", y="score", data=df_scores, ax=ax)
    st.pyplot(fig)
