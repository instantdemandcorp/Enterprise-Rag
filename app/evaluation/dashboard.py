# app/evaluation/dashboard.py
import streamlit as st, json
from app.evaluation.ragas_runner import evaluate_samples
from dotenv import load_dotenv
load_dotenv()
st.title("RAG Evaluation Dashboard (RAGAS)")
uploaded = st.file_uploader("Upload eval samples (JSONL)", type=["jsonl"])

if uploaded:
    samples = [json.loads(line) for line in uploaded.readlines()]
    with st.spinner("Evaluating..."):
        scores = evaluate_samples(samples)
    st.json(scores)

#python -m streamlit run .\app\evaluation\dashboard.py
