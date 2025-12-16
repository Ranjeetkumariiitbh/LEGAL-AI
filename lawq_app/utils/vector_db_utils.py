from typing import List
import hashlib

from langchain_community.vectorstores.chroma import Chroma


class LocalEmbeddingModel:
    """
    Simple local embedding model:
    - Koi API call nahi
    - Har text ko deterministic numeric vector me convert karta hai
    - Sirf demo / testing ke liye (koi real semantic magic nahi)
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        # Hash text -> fixed‑size vector in [-1, 1]
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals: List[float] = []
        while len(vals) < self.dim:
            for b in h:
                vals.append((b / 255.0) * 2 - 1)  # 0‑255 → -1 .. 1
                if len(vals) >= self.dim:
                    break
        return vals

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class VectorStoreHandler:
    def __init__(self) -> None:
        # Pehle yahan OpenAIEmbeddings tha – ab apni local embedding hai
        self.embedding_model = LocalEmbeddingModel()

    def create_embeddings(self, text_chunks: List[str]) -> None:
        """
        Text chunks ko embed karke Chroma DB me store karta hai.
        Koi external API use nahi hoti.
        """
        if not text_chunks:
            # Khaali list ho to Chroma ko call hi mat karo, warna error aata hai
            return

        Chroma.from_texts(
            text_chunks,
            self.embedding_model,
            persist_directory="../data/chroma_db",
        )

    def get_retriever(self):
        """
        Disk se Chroma vector store load karke retriever banata hai.
        """
        vector_store = Chroma(
            persist_directory="../data/chroma_db",
            embedding_function=self.embedding_model,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        return retriever
