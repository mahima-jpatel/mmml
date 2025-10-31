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
```

Each component is modular, allowing easy replacement of embedding or LLM backends.

---
## 3.5 Datasets Explored
To ensure that our chunking strategies generalize across **diverse textual structures**, we evaluated them on **seven large public-domain documents** with varying linguistic and organizational characteristics:

| Dataset | Description | Structure Complexity | Primary Challenges for Chunking |
|----------|-------------|----------------------|---------------------------------|
| **The Bible (KJV)** | Religious text composed of 66 books with verses, cross-references, and heavy use of pronouns (“he”, “him”, “this”). | Highly hierarchical (Book → Chapter → Verse) | Lost context & broken references between verses. |
| **The U.S. Constitution** | Foundational legal document with structured sections and amendments. | Logical hierarchy (Article → Section → Clause) | Context loss in cross-referential clauses (“as provided in the previous section…”). |
| **Moby Dick** | Literary narrative rich in metaphors and long sentences. | Linear but semantically dense | Sentence boundaries often misalign with semantic units. |
| **Les Misérables** | Philosophical and emotional narrative with internal monologues. | Multi-topic, multi-speaker structure | Semantic drift within paragraphs; requires contextual overlap. |
| **Frankenstein** | Scientific-gothic narrative with shifting first-person perspectives. | Nested narrative structure (“letters → story → story”) | Pronoun coreference; maintaining narrator identity. |
| **War and Peace** | Epic novel with historical exposition and multiple plotlines. | Long contextual dependencies across scenes | Large sections where naive chunking splits continuous reasoning. |
| **Shakespeare Complete Works** | Collection of plays and sonnets with stage directions and dialogue. | Scripted format (Act → Scene → Line) | Metadata loss and inter-speaker references. |

Processed text and evaluation prompts are available under `data/`:
```
data/
├── bible_kjv.txt
├── constitution_full.txt
├── frankenstein.txt
├── les_miserables.txt
├── moby_dick.txt
├── shakespeare_complete.txt
├── war_and_peace.txt
├── evaluation_set.jsonl
└── eval.jsonl
```

This section leads directly into the **Design Decisions** section — motivating why hierarchical and adaptive chunking became necessary to handle such a diverse corpus.

---

## 4. Design Decisions & Rationale

This project was not built as a single algorithm, but as a progressive exploration of how chunking, retrieval, and reasoning interact in RAG systems.  
Below we trace the evolution from naive heuristics to adaptive self-guided retrieval, highlighting the motivation, design trade-offs, and final insights.

#### Naive Fixed-Size Chunking
> “Split every 1000 tokens and hope context stays intact.”

**Problem:** breaks in the middle of sentences or references.  
Example:
> Chunk 1: "On February 6, 1933, Secretaryof State Henry Stimson certified tha",
> Chunk 2: "t it had become a part of the Constitution.\n",

Chunk 1 refers to something in Chunk 2 with the word "it" and retrieval fails to reconstruct meaning.

**Lesson:** token-level segmentation optimizes for compute and not comprehension.

---

#### Sentence-Based Chunking

- Uses sentence boundaries (`sent_tokenize`) to merge complete thoughts.
- Adds minimal overlap (e.g., one sentence) between adjacent chunks.

**Benefit:** syntactic completeness → better embedding coherence.  
**Limitation:** sentences ≠ semantics — consecutive sentences may still describe separate ideas.

---

#### Semantic Chunking
> "Group by meaning, not by length.”

- Uses SentenceTransformer embeddings to compute cosine similarity between sentences.
- Merges adjacent sentences until similarity falls below a threshold.

**Example:**  
Two paragraphs describing “Captain Ahab’s obsession” remain together even if long; an unrelated description of the ship’s mast forms a new chunk.

---

#### Hierarchical Chunking
> “Respect the document’s natural boundaries.”

- Builds a **three-level metadata graph**: `document → sections → paragraphs`.
- Each section stores:
  - raw text
  - paragraph list
  - auto-generated summary (`Summary:` field)
- Chunks inherit parent metadata (titles, headings).

**Benefit:**  
Retrieval can re-rank by section relevance or use summaries as lightweight surrogates when paragraphs are too fine-grained.

**Trade-off:**  
Requires up-front preprocessing of section detection and summarization.

---

### 4.2 Toward Adaptive Retrieval — Self-RAG

Even hierarchical chunking cannot guarantee that a single granularity fits all questions.  
Hence, we introduced Self-RAG, an adaptive retrieval pipeline:

1. **Start small** — retrieve the top paragraphs via hybrid BM25 + dense embeddings.  
2. **Ask an LLM** if the text directly answers the question.  
3. **If not**, escalate context:
   - include **neighboring paragraphs** (± 2)
   - expand to the **section** (with summary)
   - finally, back off to **document summary**  

This mirrors human reasoning: start local, zoom out until confidence.

**Benefits**
- Reduces hallucinated answers (retrieval stops only when the answer is confirmed).  
- Matches real-world RAG workflows that balance recall vs efficiency.  
- Enables dynamic chunk selection without re-embedding the corpus.

---

### 4.3 Key System Design Choices

| Design Choice | Rationale |
|----------------|------------|
| **Async OpenAI Calls** | Parallelized summarization and answer-checking (15 concurrent requests). |
| **Hierarchical Metadata JSON** | Provides traceability across `document_meta`, `section_meta`, and `paragraph_meta`. |
| **Hybrid Retrieval (BM25 + Dense)** | Combines lexical precision (BM25) with semantic recall (dense). |
| **Self-RAG Escalation** | Expands context progressively until the LLM confirms coverage. |
| **LLM-based Evaluation** | Measures *semantic correctness*, not just string match. |

---

### 4.4 Looking Ahead — MCTS-RAG (Next Step)

While Self-RAG explores linearly (paragraph → neighbors → section), future RAG systems can reason **non-linearly** using *search-based planning*.

#### MCTS-RAG: Monte-Carlo Tree Search for Retrieval
- Each node = a retrieval unit (chunk or section).  
- Each edge = a reasoning step (e.g., “expand context”, “refocus on entity X”).  
- **Value function** = LLM-judged answer confidence.  
- **Selection policy** = Upper Confidence Bound (UCB1) on exploration vs exploitation.  

This allows retrieval to:
- explore multiple semantic paths concurrently,  
- backtrack from misleading chunks, and  
- build *robust reasoning trees* for complex multi-hop questions.

**In short:**  
> Self-RAG is adaptive;  
> MCTS-RAG will be strategic.

---

### 4.5 Takeaway

Chunking is no longer a preprocessing step — it is an **integral part of retrieval reasoning**.  
By evolving from naive token splits to adaptive Self-RAG and planning-based MCTS-RAG, we move toward systems that **think before they fetch**.

---

## 5. Evaluation Methodology


### Evaluation Metrics



## 6. Results Summary

---

## 7. How to Run

### Setup
```bash
pip install -r requirements.txt


