"""
Two-tier frozen embedding (Sections 3.2 and practical recommendation).

Tier 1: all-MiniLM-L6-v2 (ONNX, fast screening)
Tier 2: BAAI/bge-base-en-v1.5 (confirmation on top-k)
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Manages frozen sentence-transformer models for sketch embedding."""

    def __init__(
        self,
        tier1_model: str,
        tier2_model: str,
        use_onnx: bool = True,
    ):
        self.tier1_model_name = tier1_model
        self.tier2_model_name = tier2_model
        self.use_onnx = use_onnx
        self._tier1 = None
        self._tier2 = None

    def _load_model(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        import platform

        backend = "torch"
        if self.use_onnx:
            if platform.system() == "Darwin":
                logger.info("macOS detected — using torch backend (CoreML/ONNX compatibility issues)")
            else:
                try:
                    import onnxruntime  # noqa: F401
                    backend = "onnx"
                except ImportError:
                    logger.warning("onnxruntime not installed, falling back to torch backend")

        logger.info(f"Loading model {model_name} (backend={backend})")
        model = SentenceTransformer(model_name, backend=backend)
        return model

    @property
    def tier1(self):
        if self._tier1 is None:
            self._tier1 = self._load_model(self.tier1_model_name)
        return self._tier1

    @property
    def tier2(self):
        if self._tier2 is None:
            self._tier2 = self._load_model(self.tier2_model_name)
        return self._tier2

    def embed_tier1(self, texts: list[str]) -> np.ndarray:
        """Embed sketch strings with tier-1 model. Returns (n, dim) unit vectors."""
        embeddings = self.tier1.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_tier2(self, texts: list[str]) -> np.ndarray:
        """Embed sketch strings with tier-2 model. Returns (n, dim) unit vectors."""
        prefixed = [f"Represent this sentence: {t}" for t in texts]
        embeddings = self.tier2.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False, batch_size=32,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit vectors."""
        return float(np.dot(a, b))
