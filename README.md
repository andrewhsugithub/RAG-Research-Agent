# RAG
This is a simple implementation of YouTube RAG using LlamaIndex and Qdrant or Chroma for VectorDBs.

## Usage
Replace `<query>` with your query and `<youtube_url>` with the YouTube URL.
```bash
yt-dlp -f bestaudio --extract-audio --audio-format mp3 <youtube_url> -o "audio/audio.mp3"
python whisper.py
python rag.py --query <query> --path "./qdrant" --collection "yt" --qdrant
```

## Use Cloud VectorDBs
- use [Qdrant](https://qdrant.com/) or [Pinecone](https://www.pinecone.io/) *(Note: not supported yet)*
    ```bash
    cp .env.example .env
    ```
    Copy your **QDRANT_API_KEY** and **QDRANT_URL** to the .env file


## Examples:
- hf_docs:
    Uses the [HF Docs](https://huggingface.co/datasets/hf_docs) dataset
    ```bash
    python hf_docs.py
    python rag.py --query "How to create a pipeline object?" --path "./qdrant" --collection "hf_docs" --qdrant
    ```
    See [llama3.1_hf_qdrant.txt](llama3.1_hf_qdrant.txt) for the output.

## Technologies Used:
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/)

- Embeddings (Loads from [HuggingFace](https://huggingface.co/)):
    - dense vectors: [gte-small](https://huggingface.co/thenlper/gte-small) 
    - sparse vectors: [Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1)

- VectorDBs:
    - Support Hybrid Vectors (dense + sparse)
        - [Qdrant](https://qdrant.tech/)
        - [Pinecone](https://www.pinecone.io/) *(Note: not supported yet)*
        > Note: sparse vectors defaults to [prithvida/Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1)
    - Dense Vectors: [Chroma](https://chroma.farfetch.com/)
    - Sparse Vectors: [BM25](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever)

- Reranker:
    - [bge-m3](https://huggingface.co/BAAI/bge-m3)

- Language Models (Loads from [HuggingFace](https://huggingface.co/)):
    - [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
    - [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
