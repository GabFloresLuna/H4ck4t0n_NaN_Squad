import os
import markdown
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

KB_FOLDER = "kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_FILES = 2          # máximo archivos relevantes
TIPS_PER_FILE = 3      # máximo viñetas por archivo
MIN_PARAGRAPH_LEN = 40 # descartar frases muy cortas


class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.paragraphs = []  # lista de (archivo, texto)
        self.index = None
        self.load_kb()

    def load_kb(self):
        para_texts = []
        for file in os.listdir(KB_FOLDER):
            if file.endswith(".md"):
                with open(os.path.join(KB_FOLDER, file), "r", encoding="utf-8") as f:
                    content = f.read()
                    paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > MIN_PARAGRAPH_LEN]
                    for p in paragraphs:
                        para_texts.append((file, p))
        self.paragraphs = para_texts

        embeddings = self.model.encode([p[1] for p in para_texts])
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))
        print(f"[RAG] KB cargada con {len(para_texts)} párrafos desde {len(os.listdir(KB_FOLDER))} archivos.")

    def query(self, text, k=8):
        """Busca los párrafos más similares y devuelve agrupados por archivo"""
        emb = self.model.encode([text])
        D, I = self.index.search(np.array(emb, dtype="float32"), k)

        results = {}
        for idx in I[0]:
            file, paragraph = self.paragraphs[idx]
            if file not in results:
