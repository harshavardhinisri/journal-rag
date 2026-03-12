# scripts/create_documents.py
import json
from pathlib import Path
import os

# Load dataset
with open("/Users/harsha/projects/journal-rag/data/journal_dataset_30_days.json") as f:
    journal = json.load(f)

documents = []

for entry in journal:
    date = entry["date"]
    for category in ["fitness", "work", "daily_life", "learning_upskilling"]:
        doc = {
            "text": entry[category],
            "metadata": {
                "date": date,
                "category": category
            }
        }
        documents.append(doc)

project_root = "/Users/harsha/projects/journal-rag/data"
data_dir = os.path.join(project_root, "processed")
os.makedirs(data_dir, exist_ok=True)
output_file = os.path.join(data_dir, "journal_docs.json")
# Save processed documents for embeddings
with open(output_file, "w") as f:
    json.dump(documents, f, indent=2)

print(f"Created {len(documents)} documents for RAG ingestion")