import textstat
import language_tool_python
from sentence_transformers import SentenceTransformer, util

tool = language_tool_python.LanguageTool('en-US')
sbert = SentenceTransformer("all-MiniLM-L6-v2")

def score_instruction_following(prompt, response):
    sim = util.cos_sim(sbert.encode(prompt), sbert.encode(response)).item()
    return round(sim, 2), f"Semantic similarity between prompt and response: {sim:.2f}"

def score_hallucination(response, knowledge_base):
    if response in knowledge_base:
        return 1.0, "Response found in knowledge base."
    return 0.0, "Response not found in knowledge base (possible hallucination)."

def score_assumption_control(response):
    assumptions = ["maybe", "probably", "i guess", "might"]
    penalty = sum([a in response.lower() for a in assumptions])
    score = max(0, 1 - penalty*0.25)
    return score, f"Assumption penalty: {penalty} detected."

def score_coherence_accuracy(response):
    grammar_errors = len(tool.check(response))
    readability = textstat.flesch_reading_ease(response)
    score = max(0, 1 - grammar_errors*0.05) * (readability/100)
    return round(score, 2), f"Grammar errors: {grammar_errors}, readability: {readability:.1f}"
