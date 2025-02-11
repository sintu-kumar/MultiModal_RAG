# Multimodal RAG

This project implements a multimodal retrieval-augmented generation (RAG) system. It extracts text and images from PDF documents, generates image summaries using a vision model, creates embeddings using SentenceTransformer, and stores the results in a ChromaDB vector store. A language model then uses the stored context to answer user queries.

## Features

- **PDF Content Extraction:** Uses `unstructured` to extract text and images.
- **Image Captioning:** Generates image summaries with a `blip-image-captioning-base` model.
- **Embedding Generation:** Embeds text using `SentenceTransformer` embedding model.
- **Vector Database:** Stores document elements with `ChromaDB` for retrieval.
- **Query Support:** Retrieves context and generates answers using `DeepSeek-R1-Distill-Llama-8B` model.

## Installation

- Python 3.12.2

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Place a sample PDF (e.g., `deepseekR1.pdf`) in the project directory.
2. Run the main script to process the document, store elements, display stored images, and issue a sample query:

```bash
python main.py --document_path='deepseekR1.pdf' --query='Which one, DeepSeek-R1-Zero and OpenAI o1 models AIME accuracy are better?'
```

The script will process the document, generate embeddings, and query the vector store to generate a textual response along with displaying any retrieved images.
