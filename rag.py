import os
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import faiss

# Carpeta con tus .md
KB_FOLDER = "kb"

# Modelo de embeddings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Límites de RAG
TOP_FILES = 2           # máximo archivos relevantes
TIPS_PER_FILE = 3       # máximo viñetas por archivo
MIN_PARAGRAPH_LEN = 40  # descartar frases muy cortas


class RAGEngine:
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer(EMBED_MODEL)
        self.paragraphs: List[Tuple[str, str]] = []  # [(file, text), ...]
        self.index: faiss.IndexFlatL2 | None = None
        self.load_kb()

    def load_kb(self) -> None:
        """Lee todos los .md en KB_FOLDER y construye el índice FAISS."""
        para_texts: List[Tuple[str, str]] = []

        if not os.path.exists(KB_FOLDER):
            os.makedirs(KB_FOLDER, exist_ok=True)

        md_files = [f for f in os.listdir(KB_FOLDER) if f.lower().endswith(".md")]
        for file in md_files:
            path = os.path.join(KB_FOLDER, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"[RAG] No se pudo leer {path}: {e}")
                continue

            # Split por párrafos (doble salto) y filtro de longitud
            paragraphs = [
                p.strip()
                for p in content.split("\n\n")
                if len(p.strip()) > MIN_PARAGRAPH_LEN
            ]
            for p in paragraphs:
                para_texts.append((file, p))

        self.paragraphs = para_texts

        if len(para_texts) == 0:
            print("[RAG] KB vacía o sin párrafos útiles. El índice no se construyó.")
            self.index = None
            return

        # Embeddings e indexación FAISS
        emb = self.model.encode([p[1] for p in para_texts], show_progress_bar=False)
        emb = np.asarray(emb, dtype="float32")
        dim = emb.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb)

        print(f"[RAG] KB cargada con {len(para_texts)} párrafos desde {len(md_files)} archivos.")

    def _search(self, text: str, k: int = 8) -> List[int]:
        """Retorna los índices de los párrafos más similares a 'text'."""
        if self.index is None or len(self.paragraphs) == 0:
            return []
        k = min(k, len(self.paragraphs))
        emb = self.model.encode([text], show_progress_bar=False)
        D, I = self.index.search(np.asarray(emb, dtype="float32"), k)
        return I[0].tolist()

    def query(self, text: str, max_files: int = TOP_FILES, tips_per_file: int = TIPS_PER_FILE, k: int = 16) -> Dict[str, List[str]]:
        """
        Busca párrafos relevantes y devuelve un dict:
        { "archivo.md": ["- tip 1", "- tip 2", ...], ... }
        """
        ids = self._search(text, k=k)
        if not ids:
            return {}

        # Agrupar por archivo
        grouped: Dict[str, List[str]] = {}
        for idx in ids:
            file, paragraph = self.paragraphs[idx]
            if file not in grouped:
                grouped[file] = []
            if len(grouped[file]) < tips_per_file:
                grouped[file].append(f"- {paragraph.strip()}")

        # Orden por cantidad de tips y cortar a max_files
        ordered_files = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
        return dict(ordered_files[:max_files])


def format_bullets_by_file(grouped: Dict[str, List[str]]) -> str:
    """Convierte el dict {archivo: [bullets]} en un texto legible por archivo."""
    if not grouped:
        return "• No se encontraron recomendaciones en la base de conocimiento."

    parts: List[str] = []
    for file, bullets in grouped.items():
        parts.append(f"📄 **{file}**")
        parts.extend(bullets)
        parts.append("")  # línea en blanco entre archivos
    return "\n".join(parts).strip()


# ========= Helpers de alto nivel para la API =========

# Mantener una única instancia en memoria (útil para FastAPI/Gradio)
_rag_engine: RAGEngine | None = None

def get_rag() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def get_recommendations(prompt: str, max_files: int = TOP_FILES, tips_per_file: int = TIPS_PER_FILE) -> str:
    """
    Retorna las recomendaciones formateadas en viñetas, agrupadas por archivo.
    """
    engine = get_rag()
    grouped = engine.query(prompt, max_files=max_files, tips_per_file=tips_per_file, k=24)
    return format_bullets_by_file(grouped)