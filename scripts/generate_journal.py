import requests
import json

# Start date for the journal
start_date = "2025-01-01"

# Full prompt with persona and instructions
prompt = f"""
Generate a realistic personal journal for a 26-year-old female Software Engineer living in the USA.
The journal should cover 5 consecutive days starting from {start_date}.

For each day, include four sections:
1. Fitness
2. Work
3. Daily Life
4. Learning/Upskilling

Each section should be one paragraph (4–6 sentences).
The writing should feel natural, like a real person's daily reflection, with variation and realistic events.

Return the output as a JSON array of 5 entries, like:

[
  {{
    "date": "2025-01-01",
    "fitness": "...",
    "work": "...",
    "daily_life": "...",
    "learning_upskilling": "..."
  }},
  {{
    "date": "2025-01-02",
    "fitness": "...",
    "work": "...",
    "daily_life": "...",
    "learning_upskilling": "..."
  }},
  ...
]
"""

# Send request to Ollama local API
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
)

# Extract the generated dataset
dataset = response.json()["response"]

# Optionally, save to a JSON file
with open("journal_dataset_30_days.json", "w") as f:
    f.write(dataset)

print("30-day journal dataset generated and saved!")