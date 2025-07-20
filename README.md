# QA Documentation Assistant

A command-line question answering system built with LangChain and RAG (Retrieval-Augmented Generation)

## Features
- Answers questions from documentation files
- Supports Markdown and plain text documents
- Uses HuggingFace embeddings and OpenAI LLM
- Simple command-line interface

## Quick Start

### Prerequisites
- Python 3.12.0
- OpenAI API key (for GPT models)

### Installation
- **Navigate to your projects directory (desired dir)**
```bash
cd qa-assistant
``` 
- **Clone repository**
```bash
git clone https://github.com/BarbareJ/qa-assistant.git
```

- **Create virtual environment**
```bash
python -m venv venv
```
- **Activate virtual environment**
- Linux/Mac
```bash
source venv/bin/activate
```
- Windows
```bash
.\venv\Scripts\activate
```

- **Install dependencies**
```bash
pip install -r requirements.txt
```

- **Set up environment variables**
```bash
echo 'OPENAI_API_KEY="your-api-key"' > .env
```

##  Usage
```bash
python qa.py --question "Your question here"
```

### Example
```bash
python qa.py --question "How do I reset my password?"
```
