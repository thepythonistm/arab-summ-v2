"""
Hybrid Pipeline Orchestrator:
Article → Extractive → AraT5 → Post-process
"""
from .extractive import extractive_compress
from .abstractive import AraT5Summarizer
from .postprocess import postprocess_summary


class HybridPipeline:
    def __init__(self, model_path: str, device: str = None):
        self.abstractive = AraT5Summarizer(model_path, device=device)

    def run(self, article: str, max_sentences: int = 5, max_summary_words: int = 60) -> dict:
        """
        Full pipeline execution.

        Returns:
            {
                "original_article": str,
                "extractive_output": str,
                "abstractive_summary": str,
                "final_summary": str,
                "compression_ratio": float
            }
        """
        # Step 1: Extractive compression
        key_sentences = extractive_compress(article, max_sentences=max_sentences)

        # Step 2: Abstractive generation
        raw_summary = self.abstractive.summarize(key_sentences)

        # Step 3: Post-processing
        final_summary = postprocess_summary(raw_summary, max_words=max_summary_words)

        # Metrics
        orig_len = len(article.split())
        final_len = len(final_summary.split())
        ratio = round((1 - final_len / orig_len) * 100, 2) if orig_len > 0 else 0

        return {
            "original_article": article,
            "extractive_output": key_sentences,
            "abstractive_summary": raw_summary,
            "final_summary": final_summary,
            "compression_ratio": ratio
        }