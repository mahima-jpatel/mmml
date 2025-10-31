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

Each dataset stresses the chunking system in a different way:

- **Legal and hierarchical** texts (e.g., Constitution, Bible) demand structural metadata preservation.  
- **Dialogic and narrative** texts (e.g., Shakespeare, Les Misérables) require sentence-level continuity and overlapping context.  
- **Philosophical or historical** texts (e.g., War and Peace, Frankenstein) test semantic grouping and long-range dependency modeling.

Together, these corpora provide a comprehensive benchmark for evaluating:
- chunk coherence,  
- retrieval accuracy under different granularities, and  
- robustness of contextual preservation mechanisms.

---

### Example Evaluation Files

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

This project was not built as a single algorithm, but as a **progressive exploration** of how chunking, retrieval, and reasoning interact in RAG systems.  
Below we trace the evolution from naive heuristics to adaptive self-guided retrieval — highlighting the motivation, design trade-offs, and final insights.

---

### 4.1 From Naive to Intelligent Chunking

**Goal:** make every retrieved unit of text *semantically self-contained* — so an LLM can reason over it without needing unseen context.

#### Naive Fixed-Size Chunking
> “Split every 1000 tokens and hope context stays intact.”

**Problem:** breaks in the middle of sentences or references.  
Example:
> Chunk 1: " of Representatives may choose a President whenever the right of choice shall havedevolved upon them, and for the case of the death of any of the persons from whom theSenate may choose a Vice President whenever the right of choice shall have devolved uponthem.\nSection 5\nSections 1 and 2 shall take effect on the 15th day of October following the ratification of this\narticle.\nSection 6\nThis article shall be inoperative unless it shall have been ratified as an amendment to the\nConstitution by the legislatures of three-fourths of the several States within seven yearsfrom the date of its submission.\nThe Twentieth Amendment\n17was proposed by Congress on March 2, 1932 when it passed\nthe Senate,18having previously passed the House on March 1.19It appears officially in 47 Stat.\n745. Ratification was completed on January 23, 1933, when the thirty-sixth state approved theAmendment, there being then forty-eight states in the Union. On February 6, 1933, Secretaryof State Henry Stimson certified tha",
 
> Chunk 2: "t it had become a part of the Constitution.\n20\nThe several state legislatures ratified the Twentieth Amendment on the following dates:\nVirginia, March 4, 1932; New York, March 11, 1932; Mississippi, March 16, 1932; ArkansasMarch 17, 1932; Kentucky, March 17, 1932; New Jersey, March 21, 1932; South Carolina,March 25, 1932; Michigan, March 31, 1932; Maine, April 1, 1932; Rhode Island, April 14, 1932;Illinois, April 21, 1932; Louisiana, June 22, 1932; West Virginia, July 30, 1932; Pennsylvania,August 11, 1932; Indiana, August 15, 1932; Texas, September 7, 1932; Alabama, September 13,1932; California, January 3, 1933; North Carolina, January 5, 1933; North Dakota, January 9,1933; Minnesota, January 12, 1933; Arizona, January 13, 1933; Montana, January 13, 1933;Nebraska, January 13, 1933; Oklahoma, January 13, 1933; Kansas, January 16, 1933; Oregon,January 16, 1933; Delaware, January 19, 1933; Washington, January 19, 1933; Wyoming,January 19, 1933; Iowa, January 20, 1933; South Dakota, Janu",



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


