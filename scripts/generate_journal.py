import requests
import json, os

# Start date for the journal
start_date = "2026-01-01"

# Full prompt with persona and instructions
prompt = f"""
Generate a realistic personal journal for a 26-year-old female Software Engineer living in the USA.
The journal should cover 30 consecutive days starting from {start_date}.

For each day, include four sections:
1. Fitness
2. Work
3. Daily Life
4. Learning/Upskilling

Each section should be one paragraph (8-10 sentences).
The writing should feel natural, like a real person's daily reflection, with variation and realistic events. 
The character name is Maya.
Voice: Casual, introspective, occasionally messy (use sentence fragments).
Timeline: One month (January 2026).
Arc: Start with her feeling stagnant in her junior dev role and end with her leading a small project. Start with her wasting her time on doom scrolling and then gradually cultivating hobbies like gym workouts, painting, reading.
Requirements: > * Each entry must mention at least one specific event (e.g., a meeting, a walk in the park, a bug she fixed).

Include recurring entities (a boss named Marcus, a cat named Pixel).

Vary the length; some days are just two sentences because she's tired, others are long reflections.

Return the output as a JSON array of entries, like:

[
  {{
    "date": "2026-01-01",
    "fitness": "...",
    "work": "...",
    "daily_life": "...",
    "learning_upskilling": "..."
  }},
  {{
    "date": "2026-01-02",
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

# Define project root and data directory
project_root = "/Users/harsha/projects/journal-rag"
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

# Extract the generated dataset
dataset = response.json()["response"]

# Optionally, save to a JSON file
output_path = os.path.join(data_dir, "journal_dataset_30_days.json")
with open(output_path, "w") as f:
    f.write(dataset)

print(f"30-day journal dataset generated and saved at {output_path}!")