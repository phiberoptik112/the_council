"""
ChromaDB vector store wrapper for PDF page storage and retrieval.

Uses Ollama embedding endpoint for generating embeddings locally,
keeping everything on-device for Raspberry Pi deployment.
"""
import logging
import requests
from django.conf import settings

logger = logging.getLogger('council_app.vector_store')

# Lazy-initialized ChromaDB client (singleton)
_chroma_client = None


def _get_chroma_client():
    """Get or create the ChromaDB persistent client (singleton)."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        persist_dir = str(getattr(settings, 'CHROMADB_PATH', 'chromadb_data'))
        logger.info(f"Initializing ChromaDB client at: {persist_dir}")
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def get_or_create_collection(name: str):
    """
    Get or create a ChromaDB collection by name.

    Args:
        name: Collection name (will be sanitized for ChromaDB requirements)

    Returns:
        chromadb.Collection
    """
    client = _get_chroma_client()
    # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores/hyphens
    safe_name = _sanitize_collection_name(name)
    logger.debug(f"Getting/creating collection: {safe_name}")
    return client.get_or_create_collection(name=safe_name)


def delete_collection(name: str):
    """Delete a ChromaDB collection if it exists."""
    client = _get_chroma_client()
    safe_name = _sanitize_collection_name(name)
    try:
        client.delete_collection(name=safe_name)
        logger.info(f"Deleted collection: {safe_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection '{safe_name}': {e}")


def add_page(collection_name: str, page_num: int, text: str, metadata: dict):
    """
    Add a single page's markdown text to a ChromaDB collection.

    Generates an embedding via Ollama's embedding endpoint, then stores
    the text + embedding + metadata in ChromaDB.

    Args:
        collection_name: Name of the ChromaDB collection
        page_num: Page number (used as document ID)
        text: Extracted markdown text from the page
        metadata: Dict with keys like document_id, title, filename, page_number
    """
    if not text or not text.strip():
        logger.warning(f"Skipping empty page {page_num} for collection {collection_name}")
        return

    collection = get_or_create_collection(collection_name)
    doc_id = f"page_{page_num}"

    # Generate embedding via Ollama
    embedding = _generate_embedding(text)

    if embedding:
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
    else:
        # Fall back to ChromaDB's default embedding if Ollama embedding fails
        logger.warning(f"Ollama embedding failed for page {page_num}, storing without custom embedding")
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata],
        )

    logger.debug(f"Stored page {page_num} in collection '{collection_name}'")


def search(collection_name: str, query: str, n_results: int = 5) -> list:
    """
    Search a ChromaDB collection using semantic similarity.

    Args:
        collection_name: Name of the collection to search
        query: Search query text
        n_results: Number of results to return

    Returns:
        List of dicts with keys: id, text, metadata, distance
    """
    collection = get_or_create_collection(collection_name)

    # Generate query embedding
    query_embedding = _generate_embedding(query)

    if query_embedding:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
    else:
        # Fall back to ChromaDB's default text query
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )

    # Flatten results into a simple list of dicts
    output = []
    if results and results.get('ids'):
        for i, doc_id in enumerate(results['ids'][0]):
            output.append({
                'id': doc_id,
                'text': results['documents'][0][i] if results.get('documents') else '',
                'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                'distance': results['distances'][0][i] if results.get('distances') else None,
            })

    return output


def _generate_embedding(text: str) -> list | None:
    """
    Generate an embedding vector using Ollama's /api/embed endpoint.

    Returns None if the embedding model is unavailable or fails.
    """
    config = getattr(settings, 'COUNCIL_CONFIG', {})
    pdf_config = getattr(settings, 'PDF_PROCESSING', {})
    ollama_url = config.get('OLLAMA_URL', 'http://localhost:11434')
    embedding_model = pdf_config.get('EMBEDDING_MODEL', 'nomic-embed-text')

    try:
        resp = requests.post(
            f"{ollama_url}/api/embed",
            json={
                "model": embedding_model,
                "input": text[:8000],  # Truncate to avoid oversized inputs
            },
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get('embeddings')
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
        else:
            logger.warning(
                f"Ollama embedding returned status {resp.status_code}: {resp.text[:200]}"
            )
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama for embeddings. Is Ollama running?")
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")

    return None


def _sanitize_collection_name(name: str) -> str:
    """
    Sanitize a string for use as a ChromaDB collection name.
    Requirements: 3-63 chars, starts/ends with alphanumeric,
    only contains alphanumeric, underscores, hyphens.
    """
    import re
    # Replace non-alphanumeric chars with underscores
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Strip leading/trailing underscores and hyphens
    safe = safe.strip('_-')
    # Ensure it starts with alphanumeric
    if safe and not safe[0].isalnum():
        safe = 'c_' + safe
    # Ensure minimum length
    if len(safe) < 3:
        safe = safe + '_col'
    # Truncate to max length
    if len(safe) > 63:
        safe = safe[:63].rstrip('_-')
    return safe
