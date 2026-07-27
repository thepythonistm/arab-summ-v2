"""
Hybrid Pipeline Orchestrator:
Article → Extractive → [Prompt] → AraT5 → Post-process → Fact Check
"""
import logging
from .extractive import extractive_compress
from .abstractive import AraT5Summarizer
from .postprocess import postprocess_summary
from .factcheck import fact_check

logger = logging.getLogger(__name__)


class HybridPipeline:
    def __init__(self, model_path: str, device: str = None):
        self.abstractive = AraT5Summarizer(model_path, device=device)

    def run(
        self,
        article: str,
        max_sentences: int = 5,
        extractive_max_words: int = 120,
        max_summary_words: int = 60,
        min_article_words: int = 80,
        use_instruction: bool = True,
        enable_factcheck: bool = True,
        retry_on_hallucination: bool = True
    ) -> dict:
        """Full pipeline execution with fallback + fact verification."""
        if not article or not article.strip():
            return self._empty_result(article, "Empty input")

        orig_len = len(article.split())

        # --- SHORT CIRCUIT: Skip extractive for short articles ---
        if orig_len <= min_article_words:
            logger.info("Short article: skipping extractive, direct abstractive.")
            key_text = article
        else:
            try:
                key_text = extractive_compress(
                    article,
                    max_sentences=max_sentences,
                    max_words=extractive_max_words
                )
                if not key_text or not key_text.strip():
                    key_text = article
            except Exception as e:
                logger.warning(f"Extractive failed: {e}. Using full article.")
                key_text = article

        # --- Abstractive (with instruction prompt — Step 1) ---
        try:
            raw_summary = self.abstractive.summarize(
                key_text, use_instruction=use_instruction
            )
            if not raw_summary or not raw_summary.strip():
                raise ValueError("AraT5 returned empty")
        except Exception as e:
            logger.warning(f"Abstractive failed: {e}. Falling back to extractive.")
            raw_summary = key_text

        # --- Post-process ---
        try:
            final_summary = postprocess_summary(raw_summary, max_words=max_summary_words)
        except Exception as e:
            logger.warning(f"Postprocess failed: {e}. Using raw.")
            final_summary = raw_summary

        # --- Fact Check (Step 2): against ORIGINAL article ---
        check = {"faithful": None, "fact_score": None, "unsupported_facts": []}
        retried = False

        if enable_factcheck:
            check = fact_check(final_summary, article)

            if not check["faithful"] and retry_on_hallucination:
                logger.info(
                    f"Hallucination detected ({len(check['unsupported_facts'])} facts). "
                    f"Regenerating with stricter decoding..."
                )
                try:
                    retry_summary = self.abstractive.summarize(
                        key_text, num_beams=6, use_instruction=False
                    )
                    retry_summary = postprocess_summary(retry_summary, max_words=max_summary_words)
                    retry_check = fact_check(retry_summary, article)

                    # Keep retry only if it's actually better
                    if retry_check["fact_score"] >= check["fact_score"]:
                        final_summary = retry_summary
                        check = retry_check
                        retried = True
                except Exception as e:
                    logger.warning(f"Retry failed: {e}. Keeping original.")

        # --- Metrics ---
        final_len = len(final_summary.split())
        ratio = round((1 - final_len / orig_len) * 100, 2) if orig_len > 0 else 0

        return {
            "original_article": article,
            "extractive_output": key_text,
            "abstractive_summary": raw_summary,
            "final_summary": final_summary,
            "compression_ratio": ratio,
            "skipped_extractive": orig_len <= min_article_words,
            "fallback_used": raw_summary == key_text,
            "fact_check": check,
            "regenerated": retried
        }

    def _empty_result(self, article: str, reason: str) -> dict:
        return {
            "original_article": article,
            "extractive_output": "",
            "abstractive_summary": "",
            "final_summary": "",
            "compression_ratio": 0,
            "skipped_extractive": False,
            "fallback_used": True,
            "fact_check": {"faithful": None, "fact_score": None, "unsupported_facts": []},
            "regenerated": False,
            "error": reason
        }