# ASOIAF Chatbot

Basic chatbot for A Song of Ice and Fire using text processing and embeddings.

## Features

- Text chunking with overlap
- Basic embedding search
- Configuration management with Pydantic
- Test suite

## Tech Stack

- Python 3.9+
- FastAPI
- Sentence Transformers
- ChromaDB
- Streamlit

## Usage

```bash
pip install -r requirements.txt
python src/data_processor.py --initialize
python src/data_processor.py
```

## Structure

```
├── src/
│   ├── config.py
│   └── data_processor.py
├── tests/
└── data/
```

Basic implementation for learning text processing fundamentals.
