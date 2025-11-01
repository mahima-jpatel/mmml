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
- build robust reasoning trees for complex multi-hop questions.

**In short:**  
- Self-RAG is adaptive;  
- MCTS-RAG will be strategic.

---

### 4.5 Takeaway

Chunking is no longer a preprocessing step — it is an **integral part of retrieval reasoning**.  
By evolving from naive token splits to adaptive Self-RAG and planning-based MCTS-RAG, we move toward systems that **think before they fetch**.

---
## 5. Evaluation Methodology

### 5.1 Evaluation Setup

We designed a **24-question diagnostic benchmark** to measure how each chunking strategy impacts retrieval performance in RAG.  
Each question–answer pair tests a specific retrieval failure mode:

| Category | What It Tests | Example |
|-----------|----------------|----------|
| **Understand Hierarchy** | Can the retriever follow structured hierarchy (document → section → paragraph)? | “Which section discusses the causes of global warming?” |
| **Exact Match** | Can it locate literal matches? | “Finish the line: It was the best of times, it was the ___.” |
| **Lost Context** | Can it recover multi-sentence evidence spread across chunks? | “Why did the author leave the city despite success?” |
| **Poor Retrieval** | Can it disambiguate distractors with similar keywords? | “Which law was repealed after the Boston Tea Party?” |
| **Broken References** | Can it resolve pronouns and context carried across sentences? | “Who was he referring to in the previous paragraph?” |

Each category isolates a retrieval weakness — **syntactic**, **semantic**, or **contextual** — providing a holistic view of how chunking design affects evidence recall.

---

### 5.2 Evaluation Pipeline

All chunking strategies were tested under a unified **hybrid retrieval evaluator** built on:

- **Dense retrieval** — SentenceTransformer embeddings (`all-MiniLM-L6-v2`)  
- **Sparse retrieval** — BM25 scoring for keyword overlap  
- **Hybrid fusion** — Weighted combination: `(1−α)*BM25 + α*Dense`  
- **LLM verification** — GPT-4o judges if retrieved text *explicitly* contains the answer  

We computed the following metrics:

| Metric | Definition | Purpose |
|--------|-------------|----------|
| **Exact Match** | Whether the gold answer appears verbatim in retrieved text. | Literal correctness |
| **Cosine Similarity** | Semantic similarity between retrieved and gold answers. | Embedding coherence |
| **LLM Verdict** | GPT-4o binary “YES/NO” on whether the chunk answers the query. | Semantic correctness |
| **Recall@K** | Fraction of queries with a relevant hit in top-K retrieved chunks. | Coverage |
| **MRR** | Mean Reciprocal Rank of the first correct retrieval. | Ranking quality |

---
## 6. Results & Discussion

### 6.1 Quantitative Comparison

| Method | Exact | LLM | Recall | MRR | Cosine |
|:--|:--:|:--:|:--:|:--:|:--:|
| **Naive Chunking** | 0.38 | 0.58 | 0.57 | 0.40 | 0.41 |
| **Sentence Chunking** | 0.33 | 0.62 | 0.62 | 0.48 | 0.41 |
| **Semantic Chunking** | 0.38 | 0.58 | 0.58 | 0.47 | 0.42 |
| **Hierarchical Chunking** | **0.50** | 0.58 | 0.64 | 0.49 | **0.46** |
| **Self-RAG (Hierarchical)** | **0.50** | **0.92** | **0.92** | **0.82** | 0.46 |

---

### 6.2 Qualitative Insights

#### 🧱 Naive → Sentence Chunking  
Sentence-level segmentation improves coherence and prevents mid-sentence breaks, yielding a notable rise in **MRR (0.40 → 0.48)**.  
However, Exact Match slightly drops since contiguous answer spans sometimes split across boundaries.

#### 🧩 Semantic Chunking  
Semantic grouping merges sentences that share conceptual similarity.  
This stabilizes embedding quality — even when literal overlap is absent, **cosine similarity** increases, indicating retrieval of semantically relevant chunks.

#### 🪜 Hierarchical Chunking  
Incorporating document structure (Document → Section → Paragraph) improves **Exact Match (0.50)** and **Recall@10 (0.64)**.  
This method particularly excels on *Understand Hierarchy* and *Lost Context* questions.  
Increasing fan-out (`top_docs=7`, `top_secs=50`) further improved recall by widening the search across document substructures.

#### 🔁 Self-RAG with Hierarchical Retrieval  
Self-RAG introduces a reflexive feedback loop:  
> Retrieval → LLM verification → Context expansion → Verified coverage  

This adaptive strategy dramatically boosts **LLM accuracy (0.58 → 0.92)** and **MRR (0.49 → 0.82)**, ensuring retrieval continues until the model confirms evidence sufficiency.  
In practice, this prevents “lost context” and hallucination while maintaining efficiency.

---

### 6.3 Category-wise Performance Patterns

| Category | Difficulty | Observed Trend |
|-----------|-------------|----------------|
| **Understand Hierarchy** | High | Major gains only after hierarchical modeling (MRR ↑ to 1.0). |
| **Exact Match** | Medium | Consistent across methods; robust lexical lookup. |
| **Lost Context** | High | Benefits from semantic and hierarchical chunking; fully recovered in Self-RAG. |
| **Poor Retrieval** | High | Remains challenging due to distractor overlap. |
| **Broken References** | Moderate | Improved with overlap and paragraph-neighbor expansion. |

---

### 6.4 Key Takeaways

1. **Chunking defines retrieval quality.**  
   Better chunk boundaries = higher recall and precision.  
2. **Hierarchical context restores coherence.**  
   Respecting document structure prevents fragmentation and loss of meaning.  
3. **LLM feedback closes the loop.**  
   Self-RAG transforms retrieval into a self-correcting, adaptive process.  
4. **Traditional metrics fall short.**  
   LLM-based verdicts capture semantic correctness better than literal overlap.

---

### 6.5 Summary Narrative

> Retrieval quality improves not by larger models,  
> but by **teaching models what a coherent unit of text looks like.**

Our experiments reveal a clear progression:

- **Naive Chunking** retrieves words.  
- **Sentence Chunking** retrieves thoughts.  
- **Semantic Chunking** retrieves ideas.  
- **Hierarchical Chunking** retrieves structure.  
- **Self-RAG** retrieves **understanding**.  

By the final configuration, **Self-RAG + Hierarchical Chunking** achieves  
**92 % Recall**, **0.82 MRR**, and near-perfect LLM accuracy — setting a new bar for adaptive, reasoning-aware retrieval in RAG systems.

---

## 7. How to Run

### Setup
```bash
pip install -r requirements.txt


