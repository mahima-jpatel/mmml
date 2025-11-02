# Smart Chunking for RAG
An intelligent document chunking system for improved retrieval quality in Retrieval-Augmented Generation (RAG).

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
| **Hierarchical Chunking** | Uses document structure (sections → paragraphs → sentences) and metadata to form nested chunks, each with contextual summaries. | Preserves document hierarchy and meaning. | Requires metadata extraction. |
| **Self-RAG Chunking** | Dynamically adjusts retrieval granularity using a Self-RAG loop — if a paragraph fails, expands to ±2 neighbors → section → section summary → document summary. | Emulates real retrieval behavior; robust to context loss. | Requires embeddings and LLM feedback loop. |

---

## 3. System Architecture
```
src/
├── chunkers/
│ ├── naive_chunker.py              # Basic fixed-size chunking
│ ├── smart_sentence_chunker.py     # Sentence-aware semantic chunking
│ ├── hierarchical_chunker.py       # Multi-level (para → section → doc) chunking
│
├── embeddings/
│ ├── build_hierarchical_embeddings.py  # Build multi-level FAISS + BM25 indices
│ ├── build_naive_embeddings.py         # Build flat paragraph-level indices
│ ├── build_smart_sentence_embeddings.py# Build sentence-level indices
│
├── evaluators/
│ ├── eval_config.py               # Default paths & parameters
│ ├── eval_metrics.py              # Exact Match, Cosine, MRR, Recall@K
│ ├── eval_selfrag.py              # Self-RAG evaluation (LLM verified)
│ ├── eval_unified.py              # Unified evaluator (flat/smart/hierarchical)
│ ├── eval_utils.py                # Helper functions for metrics/logging
│ ├── retrieval_flat.py            # Flat (paragraph) hybrid retrieval
│ ├── retrieval_hierarchical.py    # Hierarchical hybrid retrieval
│ ├── selfrag_retrieval.py         # Self-RAG retrieval with LLM escalation
│ ├── run_eval.py                  # Shared async/sync eval loop
│
├── summarizers/
│ ├── summarizer_utils.py          # Summary preprocessing helpers
│ ├── summarizer.py                # Generates section/doc summaries
│
├── utils/
│ ├── embed_utils.py               # Embedding creation helpers
│ ├── index_utils.py               # Build/load FAISS & BM25 indices
│ ├── io_utils.py                  # Read/write JSON + metadata
│ ├── text_utils.py                # Text cleaning & tokenization

```

Each component is modular, allowing easy replacement of embedding or LLM backends.

---
## 3.1 Datasets Explored
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

#### Contextual Summarization — The Bridge Between Sentence, Hierarchical, and Self-RAG Chunking

While sentence-based chunking improved syntactic coherence, it still lacked **contextual compression**—the ability to carry meaning upward across sections.  
To overcome this, the system introduces **multi-level summarization**, forming the core of both **Hierarchical Chunking** and **Self-RAG**.

##### a. Section-Level Summaries  
Each section is summarized **in parallel** using asynchronous OpenAI calls (up to 15 concurrent requests).  
- Every section’s paragraphs are condensed into a ≤150-token summary prefixed with *“Summary:”*.  
- These summaries act as lightweight surrogates that retain semantic meaning while reducing redundancy.  
- By embedding these summaries alongside paragraphs, retrieval can back off to summaries when detailed text is too granular.

Example:
```
{
      "sid": "doc0_sec4",
      "did": "doc0",
      "title": "The First Book of Moses: Called Genesis (cont.)",
      "summary": "In this passage from Genesis, Cain is cursed by God for killing his brother Abel, resulting in his inability to farm the land and a life of wandering. Cain expresses his distress over the severity of his punishment, fearing for his life as he becomes a fugitive. God places a mark on Cain to protect him from being killed, and Cain settles in the land of Nod, where he builds a city and has a son named Enoch, leading to a lineage that includes notable figures such as Lamech, who takes two wives and boasts about having killed a man."
}
```

##### b. Multi-Section Summaries  
After section summaries are created, the system merges related or adjacent sections and generates **tier-2 summaries**.  
This captures relationships across neighboring topics or sub-sections, such as cause-and-effect or cross-referencing between clauses.

Example:
```
{
      "mid": "doc0_sec1_multi",
      "did": "doc0",
      "title": "The First Book of Moses: Called Genesis",
      "section_ids": [
        "doc0_sec1",
        "doc0_sec2",
        "doc0_sec3",
        .....
        "doc0_sec48",
        "doc0_sec49",
        "doc0_sec50",
        "doc0_sec51"
      ],
       "summary": "In the creation narrative of Genesis, God creates the world and humanity, blessing them and establishing a day of rest. The story progresses through the fall of man, marked by Adam and Eve's disobedience, leading to Cain's punishment for murdering Abel, and ultimately to Noah's righteousness amidst humanity's wickedness, culminating in the flood and God's covenant with Noah. The narrative continues with the lineage of Abraham, highlighting God's promises and covenants, including the birth of Isaac, the destruction of Sodom and Gomorrah, and the trials faced by Abraham and his family, emphasizing themes of faith, obedience, and divine intervention."
    }
```

##### c. Document-Level Summary  
At the top of the hierarchy, all section summaries are combined into a **global document summary** (≈300–400 tokens).  
This serves as:
1. A fallback for Self-RAG escalation when no specific paragraph or section provides the answer.  
2. A semantic “fingerprint” for coarse-grained retrieval and re-ranking.

Example:
```
{
      "did": "doc0",
      "name": "bible_kjv.txt",
      "summary": "The narrative of the King James Version of the Bible, freely accessible through Project Gutenberg, spans from the creation of the world in Genesis to the establishment of a monarchy in Israel, illustrating profound themes of faith, obedience, and divine intervention. It chronicles the Israelites' journey from slavery in Egypt to the Promised Land, emphasizing the importance of community responsibility, moral conduct, and the consequences of straying from God's commands. Through the lives of key figures such as Moses, Joshua, and David, the text underscores the complexities of leadership and the enduring relationship between God and His people, ultimately reminding readers of the necessity of repentance and gratitude as they navigate their spiritual journey."
    }
```

##### d. Why Summaries Matter  
Summaries transform the hierarchy from static structure into a **semantic pyramid**:  
- **Bottom:** full paragraphs preserve details.  
- **Middle:** section and multi-section summaries compress local meaning.  
- **Top:** document summary captures global context.  

During Self-RAG, retrieval naturally climbs this pyramid — starting local, escalating through summaries, and finally using document-level context if needed.  
This makes retrieval **adaptive, scalable, and meaning-aware** without re-embedding or re-chunking text.


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

### 4.4 Takeaway

Chunking is no longer a preprocessing step — it is an **integral part of retrieval reasoning**.  
By evolving from naive token splits to adaptive Self-RAG, we move toward systems that **think before they fetch**.

---
## 5. Evaluation Methodology

### 5.1 Evaluation Setup

We designed a **24-question diagnostic benchmark** to measure how each chunking strategy impacts retrieval performance in RAG.  
Each question–answer pair tests a specific retrieval failure mode:

| Category | What It Tests | Example |
|-----------|----------------|----------|
| **Understand Hierarchy** | Can the retriever follow structured hierarchy (document → section → paragraph)? | “In which book of the bible is the story of David and Goliath found?” |
| **Exact Match** | Can it locate literal matches? | “What is the color of the whale in Moby Dick?” |
| **Lost Context** | Can it recover multi-sentence evidence spread across chunks? | “How does Frankenstein describe the moment he saw the creature?” |
| **Poor Retrieval** | Can it disambiguate distractors with similar keywords? | “What does the U.S. Constitution establish as the supreme law of the land?” |
| **Broken References** | Can it resolve pronouns and context carried across sentences? | “In War and Peace, who is 'he' when it says 'he gazed at the battlefield'?” |

Each category isolates a retrieval weakness — **syntactic**, **semantic**, or **contextual**, providing a holistic view of how chunking design affects evidence recall.

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


LLM Judge Prompt
```
messages = [
            {"role": "system", "content": (
                "You are a strict retrieval evaluator. "
                "Reply YES if the retrieved text explicitly contains the expected answer words; "
                "otherwise reply NO.")},
            {"role": "user", "content": f"Question: {query}\nExpected Answer: {answer}\nText:\n{text}"},
        ]
```

---
## 6. Results & Discussion

### 6.1 Quantitative Comparison

| Method | Exact | LLM | Recall | MRR | Cosine |
|:--|:--:|:--:|:--:|:--:|:--:|
| **Naive Chunking** | 0.33 | 0.54 | 0.53 | 0.41 | 0.42 |
| **Sentence Chunking** | 0.38 | 0.58 | 0.58 | 0.43 | 0.43 
| **Hierarchical Chunking** | **0.50** | 0.54 | 0.64 | 0.54 | **0.46** |
| **Self-RAG (Hierarchical)** | **0.50** | **0.92** | **0.92** | **0.82** | 0.46 |

---

### 6.2 Qualitative Insights

#### Naive → Sentence Chunking  
Sentence-level segmentation improves coherence and prevents mid-sentence breaks, yielding a notable rise in **MRR (0.40 → 0.48)**.  
However, Exact Match slightly drops since contiguous answer spans sometimes split across boundaries.

#### Hierarchical Chunking  
Incorporating document structure (Document → Section → Paragraph) improves **Exact Match (0.50)** and **Recall@10 (0.64)**.  
This method particularly excels on *Understand Hierarchy* and *Lost Context* questions.  
Increasing fan-out (`top_docs=7`, `top_secs=50`) further improved recall by widening the search across document substructures.

#### Self-RAG with Hierarchical Retrieval and Summarization  
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
   LLM-based verdicts capture semantic correctness better than literal overlap

By the final configuration, **Self-RAG + Hierarchical Chunking and Summarization** achieves  
**92 % Recall**, **0.82 MRR**, and near-perfect LLM accuracy, setting a new bar for adaptive, reasoning-aware retrieval in RAG systems.

---

### Environment Setup

1. **Create and activate a virtual environment**

```bash
conda create -n smartchunk python=3.9
conda activate smartchunk
pip install -r requirements.txt
```

2. **Upgrade pip and essential build tools**
```bash
pip install --upgrade pip setuptools wheel
```

3. **Install all dependencies**
```bash
pip install --upgrade pip setuptools wheel
```

4. **Set up your OpenAI API key**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

5. **Preparing Artifacts: Before running evaluations, ensure that your embeddings and metadata are built for the corpus**
   
   A. Naive Artifacts
   ```bash
   python -m src.embeddings.build_naive_embeddings
   ```
   This will generate:
   ```
   artifacts/naive_faiss/
   ├── faiss_flat.index
   ├── bm25_index.pkl
   ├── metadata.json
   └── paragraphs.json

   ```

   B. SmartSentence (Sentence-level Semantic Index)
   ```bash
   python -m src.embeddings.build_smart_sentence_embeddings
   ```

   C. Hierarchical Artifacts
   ```bash
   python -m src.embeddings.build_hierarchical_embeddings
   ```

   D. Verify and Summarize Metadata
   ```bash
   python -m src.summarizers.summarizer --meta_path artifacts/hierarchical_faiss/metadata.json
   ```

6. **Running Evaluations: After preparing artifacts, run evaluations for each retrieval mode**
   
   A. Naive
   ```bash
   python -m src.evaluators.eval_unified --artifacts artifacts/naive_faiss
   ```
   B. Smart Sentence
   ```bash
   python -m src.evaluators.eval_unified --artifacts artifacts/smart_sentence_faiss
   ```
   C. Hierarchical
   ```bash
   python -m src.evaluators.eval_unified --artifacts artifacts/hierarchical_faiss
   ```
   D. Self-Rag
   ```bash
   python -m src.evaluators.eval_selfrag --top_k 5 --alpha_dense 0.5
   ```

### Reproducibility Notes:
- Embedding model: all-MiniLM-L6-v2
- FAISS index: IndexFlatIP (inner product)
- BM25: rank_bm25.BM25Okapi with simple whitespace/lowercase tokenizer; no stemming/stopword removal.
- Hybrid scoring: score = (1 - α) * BM25 + α * cosine_dense with α = 0.5 unless stated.
- LLM judge (Self-RAG): gpt-4o

---

## 7. Next Steps — MCTS-RAG (Not Implemented Yet)

While **Self-RAG** adaptively follows a single reasoning path, moving linearly from paragraph → neighbors → section, the next generation of retrieval systems will reason **non-linearly**, exploring multiple possible information paths simultaneously.

### MCTS-RAG: Monte-Carlo Tree Search for Retrieval

**Core Idea:**  
Treat retrieval as a **search problem** rather than a fixed pipeline.  
Each node in the tree represents a retrieved chunk or section, and each edge corresponds to a reasoning or query expansion step.

- **Node = retrieval unit:** a paragraph, section, or retrieved snippet.  
- **Edge = reasoning move:** e.g., “expand around this topic,” “narrow focus to entity X,” or “follow a causal relationship.”  
- **Value function =** measures how useful a chunk is for answering the question (judged by an LLM or scoring model).  
- **Selection policy =** an **Upper Confidence Bound (UCB1)** balancing exploration (new paths) and exploitation (known relevant ones).

---

### Example: Query Expansion in Action

**User Query:**  
> “How does deforestation in the Amazon impact global rainfall?”

1. **Root Node:** Retrieve chunks directly matching “deforestation in Amazon.”  
2. **Expansion Step:** Generate new, semantically related queries such as:  
   - “Amazon evapotranspiration and rainfall patterns”  
   - “carbon cycle effects on precipitation”  
   - “regional climate feedback loops”  
3. **Simulation (Rollout):** Explore each expanded query, retrieve evidence, and estimate confidence scores for how well the content supports the answer.  
4. **Backpropagation:** Propagate scores up the tree, prioritizing promising reasoning paths while pruning irrelevant or redundant ones.

Over multiple rollouts, MCTS-RAG doesn’t just retrieve, it plans its retrieval, discovering new, contextually relevant query expansions dynamically.

---

### Why It Matters

| Concept | Self-RAG | MCTS-RAG |
|----------|-----------|-----------|
| **Retrieval Strategy** | Adaptive and sequential | Strategic and exploratory |
| **Reasoning** | Follows one path at a time | Explores multiple reasoning paths concurrently |
| **Query Expansion** | Local and reactive | Global and planned via search |
| **Ideal Use Case** | Focused, short reasoning chains | Complex, multi-hop, cross-domain reasoning |

**In short:**  
- Self-RAG is adaptive.  
- MCTS-RAG will be strategic — combining retrieval and reasoning through deliberate, tree-based exploration.

---


