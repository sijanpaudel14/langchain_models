# LangChain Models Learning Repository

A comprehensive collection of examples and implementations showcasing various LangChain integrations with different AI models and embedding techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Chat Models](#chat-models)
- [Embedding Models](#embedding-models)
- [LLM Models](#llm-models)
- [Key Concepts Learned](#key-concepts-learned)
- [Usage Examples](#usage-examples)
- [API Keys Required](#api-keys-required)
- [Dependencies](#dependencies)

## 🎯 Overview

This repository demonstrates practical implementations of various Large Language Models (LLMs) and embedding models using LangChain framework. It covers multiple AI providers including OpenAI, Anthropic, Google, and Hugging Face, with both cloud-based and local model examples.

## 📁 Project Structure

```
langchain_models/
├── README.md
├── requirements.txt
├── test.py
├── ChatModels/
│   ├── 1_chatmodel_openai.py          # OpenAI GPT-4 chat implementation
│   ├── 2_chatmodel_Claudanthropic.py  # Anthropic Claude integration
│   ├── 3_chatmodel_google.py          # Google Gemini integration
│   ├── 4_chatmodel_hf.py              # Hugging Face cloud models
│   └── 5_chatmodel_hf_local.py        # Local Hugging Face models
├── EmbeddedModels/
│   ├── 1_embedding_openai_query.py    # OpenAI embeddings for queries
│   ├── 2_embedded_openai_docs.py      # OpenAI embeddings for documents
│   ├── 3_embedded_hf_local.py         # Local Hugging Face embeddings
│   ├── 4_document_similarity.py       # Document similarity with OpenAI
│   └── 5_document_similarity_local.py # Document similarity with local models
├── LLMs/
│   └── 1_lll_demo.py                  # Basic LLM demonstration
└── langenv/                           # Virtual environment
```

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd langchain_models
```

### 2. Create Virtual Environment

```bash
python -m venv langenv
source langenv/bin/activate  # On Linux/Mac
# or
langenv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

## 💬 Chat Models

### 1. OpenAI Chat Model (`ChatModels/1_chatmodel_openai.py`)

- **Model**: GPT-4
- **Features**: Temperature control, token limits
- **Key Learning**: Understanding temperature settings for creativity vs determinism

**Temperature Guidelines:**

- `0.0 - 0.3`: Deterministic, factual (math, code)
- `0.4 - 0.7`: Balanced (general Q&A, explanations)
- `0.8 - 1.2`: Creative (storytelling, jokes)
- `1.3 - 2.0`: Maximum creativity (brainstorming, wild ideas)

### 2. Anthropic Claude (`ChatModels/2_chatmodel_Claudanthropic.py`)

- **Model**: Claude-3.5-Sonnet
- **Features**: Advanced reasoning capabilities
- **Use Case**: Complex conversations and analysis

### 3. Google Gemini (`ChatModels/3_chatmodel_google.py`)

- **Model**: Gemini-1.5-Flash
- **Features**: Multimodal capabilities
- **Use Case**: Fast, efficient responses

### 4. Hugging Face Cloud (`ChatModels/4_chatmodel_hf.py`)

- **Model**: Zephyr-7B-Beta
- **Features**: Open-source model via API
- **Use Case**: Cost-effective chat solutions

### 5. Hugging Face Local (`ChatModels/5_chatmodel_hf_local.py`)

- **Model**: DistilGPT2
- **Features**: Runs locally, no API required
- **Use Case**: Privacy-focused applications

## 🔧 Embedding Models

### 1. OpenAI Embeddings - Query (`EmbeddedModels/1_embedding_openai_query.py`)

- **Model**: text-embedding-3-large
- **Dimensions**: 32 (customizable)
- **Purpose**: Convert text queries to vector representations

### 2. OpenAI Embeddings - Documents (`EmbeddedModels/2_embedded_openai_docs.py`)

- **Features**: Batch document embedding
- **Use Case**: Document processing and indexing

### 3. Local Hugging Face Embeddings (`EmbeddedModels/3_embedded_hf_local.py`)

- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Features**: Local processing, no API costs
- **Use Case**: Offline embedding generation

### 4. Document Similarity - OpenAI (`EmbeddedModels/4_document_similarity.py`)

- **Features**: Cosine similarity calculation
- **Use Case**: Finding most relevant documents
- **Example**: Cricket player information retrieval

### 5. Document Similarity - Local (`EmbeddedModels/5_document_similarity_local.py`)

- **Features**: Same functionality with local models
- **Benefit**: No API costs, privacy preservation

## 🤖 LLM Models

### Basic LLM Demo (`LLMs/1_lll_demo.py`)

- **Model**: GPT-3.5-turbo-instruct
- **Purpose**: Demonstrates basic LLM usage
- **Use Case**: Simple text completion tasks

## 🎓 Key Concepts Learned

### 1. **Model Types**

- **Chat Models**: Conversational interfaces (ChatOpenAI, ChatAnthropic)
- **LLM Models**: Text completion models (OpenAI)
- **Embedding Models**: Vector representation generators

### 2. **Temperature Control**

- Controls randomness/creativity in responses
- Lower values = more deterministic
- Higher values = more creative

### 3. **Token Management**

- Tokens ≈ words
- `max_completion_tokens`: Limits response length
- Important for cost control

### 4. **Embedding Dimensions**

- Higher dimensions = more detailed representations
- Trade-off between accuracy and computational cost

### 5. **Similarity Metrics**

- **Cosine Similarity**: Measures vector similarity (0-1)
- Used for document retrieval and recommendation systems

### 6. **Local vs Cloud Models**

- **Cloud**: Better performance, API costs
- **Local**: Privacy, no ongoing costs, offline capability

## 📝 Usage Examples

### Quick Start - Chat Model

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4", temperature=0.7)
result = model.invoke("What is the capital of Nepal?")
print(result.content)
```

### Quick Start - Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
vector = embedding.embed_query("Your text here")
```

### Document Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Get embeddings for documents and query
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate similarity scores
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find most similar document
best_match = max(enumerate(scores), key=lambda x: x[1])
```

## 🔑 API Keys Required

- **OpenAI**: For GPT models and OpenAI embeddings
- **Anthropic**: For Claude models
- **Google**: For Gemini models
- **Hugging Face**: For cloud-based Hugging Face models

## 📦 Dependencies

### Core LangChain

- `langchain`: Main framework
- `langchain-core`: Core functionality

### Model Integrations

- `langchain-openai`: OpenAI integration
- `langchain-anthropic`: Anthropic integration
- `langchain-google-genai`: Google integration
- `langchain-huggingface`: Hugging Face integration

### ML Libraries

- `transformers`: Hugging Face transformers
- `scikit-learn`: Machine learning utilities
- `numpy`: Numerical computations

### Utilities

- `python-dotenv`: Environment variable management
- `huggingface-hub`: Hugging Face model hub

## 🚀 Running the Examples

1. **Activate virtual environment**:

   ```bash
   source langenv/bin/activate
   ```

2. **Set up environment variables** in `.env` file

3. **Run any example**:
   ```bash
   python ChatModels/1_chatmodel_openai.py
   python EmbeddedModels/4_document_similarity.py
   ```

## 🎯 Next Steps

- Explore advanced LangChain features (chains, agents, tools)
- Implement RAG (Retrieval Augmented Generation) systems
- Build conversational AI applications
- Experiment with fine-tuning local models
- Create production-ready applications with proper error handling

## 📄 License

This project is for educational purposes. Please respect the terms of service of all AI providers used.

---

**Happy Learning! 🚀**

For questions or improvements, feel free to contribute or reach out.
