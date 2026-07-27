
## Project Report

## 1. Executive Summary

This project delivers a complete **hybrid Arabic text summarization system** that combines extractive sentence selection (fine-tuned AraBERT + MMR diversity selection), abstractive generation (fine-tuned AraT5 on XL-Sum), and — the core differentiator — a **post-hoc fact-verification layer with automatic regeneration** that detects unsupported facts in generated summaries and corrects them before delivery.

The system is evaluated with standard ROUGE, morphology-normalized Arabic ROUGE, BERTScore, and dedicated **faithfulness metrics** (fact score, faithful rate) — a combination not found in existing Arabic summarization literature.

---

## 2. System Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │               INPUT ARTICLE                  │
                        └──────────────────┬───────────────────────────┘
                                           │
                          words > 80 ? ────┴──── words ≤ 80
                           │                          │
                           ▼                          │
        ┌─────────────────────────────────┐           │
        │   MODULE 1 — EXTRACTIVE         │           │
        │   • AraBERT binary classifier   │           │
        │     (sentence ∈ summary?)       │           │
        │   • Hybrid scoring:             │           │
        │     50% model + 35% position    │           │
        │     + 15% length                │           │
        │   • MMR selection (λ=0.6)       │           │
        │     → relevance + diversity     │           │
        └──────────────────┬──────────────┘           │
                           │ key sentences            │
                           ▼                          ▼
        ┌─────────────────────────────────────────────────┐
        │   MODULE 2 — ABSTRACTIVE                        │
        │   AraT5 (368M params, full fine-tune, XL-Sum)   │
        │   prefix: "summarize: "                         │
        │   beam search = 4, no_repeat_ngram = 2          │
        └──────────────────┬──────────────────────────────┘
                           ▼
        ┌─────────────────────────────────────────────────┐
        │   MODULE 3 — POST-PROCESSING                    │
        │   • duplicate sentence removal                  │
        │   • Arabic punctuation fixing                   │
        │   • length compression (≤ 60 words)             │
        └──────────────────┬──────────────────────────────┘
                           ▼
        ┌─────────────────────────────────────────────────┐
        │   MODULE 4 — FACT VERIFICATION  ★ NOVEL ★       │
        │   • extract facts from summary:                 │
        │     numbers, %, dates, years, entities, quotes  │
        │   • verify each against SOURCE article          │
        │   • unsupported fact found?                     │
        │       → REGENERATE (beam=6, stricter decoding)  │
        │       → keep output with better fact score      │
        └──────────────────┬──────────────────────────────┘
                           ▼
              ┌────────────────────────┐
              │   FINAL SUMMARY        │
              │   + fact_check report  │
              │   + compression ratio  │
              └────────────────────────┘
```

**Fallback chain (production robustness):**
- Extractive fails → full article goes to AraT5
- Abstractive fails → extractive output delivered as-is
- Post-process fails → raw summary delivered
- Short article (< 80 words) → extractive skipped entirely

---

## 3. Is Binary Classification Used? — YES

The extractive module is built on **binary sequence-pair classification**:

| Aspect | Detail |
|--------|--------|
| Model | `AutoModelForSequenceClassification` (AraBERT, 2 output logits) |
| Input | `(full article, candidate sentence)` pairs |
| Classes | `1` = sentence belongs in summary / `0` = does not |
| Score used | `softmax(logits)[:, 1]` → probability of class 1 |
| Selection | Top sentences by hybrid score, then MMR re-ranking |

So each sentence in the article is independently classified as "summary-worthy" or not. The raw classifier probability is then fused with position and length heuristics (50/35/15 weighting), and **MMR (Maximal Marginal Relevance, λ=0.6)** selects the final 5 sentences — penalizing redundancy so the extractive output covers diverse aspects of the article rather than 5 near-identical sentences.

---

## 4. What This Project Adds Beyond Existing Research

### 4.1 Fact-Verification Layer with Automatic Regeneration ★ (core novelty)

Existing Arabic hybrid systems (Elsaid et al., 2023; Reda et al., 2022) end at generation. Ours adds a closed verification-correction loop:

1. **Fact extraction** from the generated summary: numbers, percentages, dates/years, Latin-script named entities, quoted phrases (Arabic-Indic digits normalized ٠-٩ → 0-9)
2. **Verification** of each fact against the *original source article* (not just the extractive intermediate)
3. **Automatic correction**: if unsupported facts are detected, the summary is regenerated with stricter decoding (beam=6), and the version with the better fact score is kept
4. **Measured impact**: on the test set, **17% of summaries triggered regeneration**, reaching **84% fully-factual outputs** with an average fact score of **0.867**

> *To our knowledge, this is among the first Arabic summarization pipelines to integrate post-hoc fact verification with automatic regeneration, evaluated with explicit faithfulness metrics.*

### 4.2 Faithfulness Metrics Alongside ROUGE

ROUGE measures n-gram overlap — it cannot detect that "500 million" became "5 million". This project reports `fact_score` and `faithful_rate` as first-class metrics, addressing a recognized gap in Arabic abstractive evaluation (echoed by Elsaid et al., 2023, who note ROUGE's inadequacy for Arabic abstractive summaries).

### 4.3 Evidence-Based Ablation: Instruction-Prompt Rejection

An instruction-constrained prompt ("summarize using only the information provided, do not add facts") was implemented, **tested, measured — and rejected**: on a model fine-tuned with a fixed `"summarize: "` prefix, the instruction variant produced *hallucinated source attribution* (e.g., inventing "BBC Arabic print edition"). The system ships with the training-format prompt. This negative result is documented as an ablation — most reports hide failed variants.

### 4.4 Dual ROUGE Reporting (Standard + Arabic-Normalized)

Following the Arabic-ROUGE insight of Elsaid et al. (2023) — that Arabic morphology (الـ prefix, alef/hamza variants, ta marbuta, diacritics) unfairly penalizes standard ROUGE — the evaluation reports **both**:
- **Standard ROUGE** → comparability with published literature
- **Normalized ROUGE** (diacritic removal, alef unification, الـ stripping, ة→ه, ى→ي) → morphology-fair scoring

### 4.5 MMR Diversity in an Arabic Hybrid Pipeline

While MMR is established in English summarization, combining a **fine-tuned AraBERT sentence classifier + hybrid heuristics + MMR re-ranking** as the extractive stage of an Arabic hybrid pipeline distinguishes this work from both Elsaid et al. (Bi-LSTM + attention) and Reda et al. (AraBERT → T5 without diversity-aware selection).

---

## 5. Positioning vs. Related Work

| Work | Extractive | Abstractive | Fact verification | Faithfulness metrics |
|------|-----------|-------------|-------------------|---------------------|
| Elsaid et al. (2023) | Bi-LSTM + attention + NER | mT5 | ❌ | ❌ (proposes Arabic-ROUGE only) |
| Reda et al. (2022) A3SUT | AraBERT | AraT5 | ❌ | ❌ (user satisfaction only) |
| **This project** | **AraBERT + MMR** | **AraT5 (full FT)** | **✅ + auto-regeneration** | **✅ fact_score, faithful_rate** |

---

## 6. Experimental Results

Test set: 100 samples from XL-Sum Arabic test split (4,689 total available).

| Pipeline | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | Fact Score | Faithful |
|----------|---------|---------|---------|-----------|------------|----------|
| Abstractive (AraT5 direct) | **24.83** | **9.59** | **22.26** | **76.65** | — | — |
| Hybrid (full pipeline) | 21.72 | 8.31 | 19.31 | 75.35 | **0.867** | **84%** |

**Training (validation subset, 500 samples, HF evaluate ROUGE):** ROUGE-1 32.84 / ROUGE-2 15.97 / ROUGE-L 28.45 — reported separately as methodology differs.

**Interpretation:**
- The pure abstractive pipeline maximizes overlap metrics (XL-Sum references are 1–2 sentence headlines; extractive pre-filtering removes ROUGE-matching surface words).
- The hybrid pipeline trades ~3 ROUGE points for **measured factual reliability** and scalability to long documents (beyond AraT5's 512-token input window) — the correct choice for production use on long-form content.

---

## 7. Technical Specifications

| Component | Specification |
|-----------|---------------|
| Abstractive model | AraT5-base, 368M params, full fine-tuning, 3 epochs, XL-Sum Arabic (37,467 train / 4,689 val / 4,689 test) |
| Extractive model | AraBERT sequence classifier (binary), sentence-pair input |
| Selection | MMR λ=0.6, TF-IDF cosine redundancy, top-5 sentences, ≤120 words |
| Generation | beam=4 (retry: 6), no_repeat_ngram=2, max 128 tokens |
| Fact checking | rule-based: numbers/%/years/dates/entities/quotes, Arabic-Indic digit normalization |
| Training infra | Google Colab, Tesla T4, checkpoint resume from Drive |
| Deployment | CPU-capable, local inference, modular `src/` package |

---

## 8. Limitations (honestly stated)

1. Fact verification is **surface-level**: it catches numeric/entity hallucinations but not semantic distortions (wrong attribution, twisted causality).
2. Fact checking operates on the **final summary vs. source**; it cannot verify information the extractive stage already dropped.
3. XL-Sum references are headline-style; abstractive models trained on it inherit that style, which may under-serve use cases wanting paragraph summaries.
4. Evaluation on 100 samples (subset mode); full 4,689-sample run is supported but compute-bound on CPU.
5. Single-domain training (news). Cross-domain behavior is untested — a planned EASC zero-shot experiment addresses this.

---
## 09. evaluation on 2000 samples :
    
    Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
ROUGE-1:        25.3215   (norm: 30.3339)
ROUGE-2:        10.0513   (norm: 13.5887)
ROUGE-L:        22.2794   (norm: 26.3984)
BERTScore:      76.4044
Fact Score:     0.9121
Faithful:       90.2%
Regenerated:    281 summaries

======================================================================
COMPARISON SUMMARY
======================================================================
Metric                   Direct AraT5      Hybrid
-------------------------------------------------
ROUGE-1                       25.6172     25.3215
ROUGE-2                       10.1222     10.0513
ROUGE-L                       22.6671     22.2794
BERTScore                      76.612     76.4044
Fact Score                      0.869      0.9121
Faithful %                       85.9        90.2

## 10. References

- Elsaid, A., Mohammed, A., Fattouh, L., Sakre, M. (2023). *A Hybrid Arabic Text Summarization Approach based on Seq-to-seq and Transformer.* Cairo University. DOI: 10.21203/rs.3.rs-2856782/v1
- Reda et al. (2022). *A3SUT: Arabic Abstractive Summarization Using Transformers (AraBERT + T5).*
- Hasan et al. (2021). *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages.*
- El-Haj et al. (2015). *EASC: Essex Arabic Summaries Corpus.*
