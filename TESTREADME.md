# Smart Chunking for RAG
*An intelligent document chunking system for improved retrieval quality in Retrieval-Augmented Generation (RAG).*

---

## 1. Overview

Retrieval-Augmented Generation (RAG) systems rely on breaking long documents into smaller “chunks” that fit within model context limits. However, naive chunking (e.g., splitting text every N tokens) often leads to loss of meaning, broken references, and reduced retrieval accuracy.

This project presents a **modular and efficient smart chunking pipeline** that tackles four major problems:
- **Lost Context:** Ensures chunks never start mid-sentence.
- **Broken References:** Preserves forward and backward references using overlap and contextual summaries.
- **Poor Retrieval:** Groups semantically coherent units to align retrieval with meaning.
- **Missing Metadata:** Embeds document hierarchy (titles, sections, headings) into chunk text.

The final system combines **four chunking strategies** with a **Self-RAG evaluation loop** to demonstrate improvements over naive methods.

---

## 2. Implemented Chunking Strategies

| Strategy | Description | Pros | Limitations |
|-----------|--------------|------|--------------|
| **Naive Chunking** | Splits text into fixed-size token windows (e.g., 500 tokens). | Simple and fast. | Breaks mid-sentence; no awareness of structure. |
| **Sentence-based Chunking** | Uses NLTK or spaCy to segment text into complete sentences, then merges them up to a token threshold. | Preserves syntactic coherence. | Ignores deeper semantic connections. |
| **Semantic Chunking** | Uses a SentenceTransformer model to group adjacent sentences with high cosine similarity (semantic coherence). | Retains meaning and context. | Slightly more compute-heavy. |
| **Hierarchical Chunking** | Uses document structure (sections → paragraphs → sentences) and metadata to form nested chunks, each with contextual summaries. | Preserves document hierarchy and meaning. | Requires metadata extraction. |
| **Self-RAG Chunking** | Dynamically adjusts retrieval granularity using a Self-RAG loop — if a paragraph fails, expands to ±2 neighbors → section → section summary → document summary. | Emulates real retrieval behavior; robust to context loss. | Requires embeddings and LLM feedback loop. |

---

## 3. System Architecture
```
src/
├── chunkers/
│ ├── naive_chunker.py
│ ├── sentence_chunker.py
│ ├── semantic_chunker.py
│ ├── hierarchical_chunker.py
│
├── metadata/
│ ├── build_metadata.py # builds section, paragraph maps
│ ├── summarize_sections.py # generates summaries for hierarchy
│
├── evaluation/
│ ├── eval_retrieval.py # computes metrics (Exact, ROUGE, Cosine, LLM)
| ├── self_rag.py


Each component is modular, allowing easy replacement of embedding or LLM backends.

---

## 4. Design Decisions

| Design Choice | Rationale |
|----------------|------------|
| **Async OpenAI Calls** | Used for efficient summarization of large datasets (15 concurrent requests). |
| **Hierarchical Metadata JSON** | Keeps `document_meta`, `section_meta`, and `paragraph_meta` layers for traceability. |
| **Hybrid Retrieval (BM25 + Dense)** | Balances lexical precision with semantic recall. |
| **Self-RAG Escalation** | Mimics real-world RAG behavior — gradually widens search until a confident answer is found. |
| **LLM-based Evaluation** | Goes beyond exact string match — measures semantic correctness using GPT-4o-mini. |

---

## 5. Evaluation Methodology

### Dataset
Three public-domain texts were used:
1. *Moby Dick* (literary narrative)
2. *The Bible (KJV)* (hierarchical structure)
3. *The U.S. Constitution* (legal document)

Each document was processed into chunks via all four strategies.

### Evaluation Metrics

| Metric | Description |
|--------|--------------|
| **Exact Match** | Binary match of gold answer text. |
| **ROUGE-L F1** | Overlap-based similarity score. |
| **Cosine Similarity** | Semantic closeness of embeddings. |
| **LLM Judge** | GPT-4o-mini verdict on whether retrieved text answers query. |
| **Mean Score** | Averaged composite metric. |

### Self-RAG Query Categories

- *Exact Match*
- *Lost Context*
- *Broken References*
- *Poor Retrieval*
- *Understand Hierarchy*

---

## 6. Results Summary

| Chunking Strategy | Mean Retrieval Score | LLM Correct (%) |
|--------------------|---------------------|-----------------|
| Naive (500 tokens) | 0.18 | 32% |
| Sentence-based | 0.31 | 54% |
| Semantic | 0.42 | 67% |
| Hierarchical | 0.52 | 73% |
| **Self-RAG (proposed)** | **0.66** | **88%** |

Self-RAG achieved nearly **3× improvement over naive chunking**, particularly in the *“Lost Context”* and *“Understand Hierarchy”* categories.

---

## 7. How to Run

### Setup
```bash
pip install -r requirements.txt


