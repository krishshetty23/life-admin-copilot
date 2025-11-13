# Life Admin Copilot

An AI assistant that turns bureaucratic emails into actionable tasks.

## Day 1: Personal Knowledge Base
- Built searchable profile using embeddings
- Implemented semantic search over personal data
- Compared keyword search vs. embedding-based search

## Tech Stack
- Python
- sentence-transformers
- NumPy

## Files
- `myProfile.txt` - Personal information knowledge base
- `search_v1.py` - Basic keyword search (before AI)
- `search_v2.py` - Semantic search with embeddings (AI-powered)

## How to Run
```bash
# Install dependencies
pip install sentence-transformers numpy

# Run keyword search
python search_v1.py

# Run semantic search
python search_v2.py
```

## Next Steps
- Email parsing with LLMs
- Automated reply generation
- Full RAG pipeline

---

*Built as part of a 6-day intensive AI/ML learning project*
