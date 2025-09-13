import pandas as pd
import random

def generate_synthetic_data(n=100):
    prompts = [
        "Summarize this text",
        "Translate to French",
        "List 3 pros and cons",
        "Explain in simple terms"
    ]
    responses = [
        "Here is a correct and concise answer.",
        "This might be true but I guess not sure.",
        "Totally unrelated hallucinated response.",
        "Translated: Bonjour, comment ça va?"
    ]
    data = []
    for i in range(n):
        data.append({
            "agent_id": f"agent_{random.randint(1,10)}",
            "prompt": random.choice(prompts),
            "response": random.choice(responses)
        })
    return pd.DataFrame(data)
