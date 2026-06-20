# Enterprise Compliance Audit System with Advanced Multi-Stage RAG

A robust, production-ready compliance auditing application leveraging an advanced Retrieval-Augmented Generation (RAG) pipeline to analyze high-volume unstructured corpora. This process-driven application operates continuously in the background to execute automated policy verification, features strict epistemic token boundaries, and implements localized data failover guardrails.

## ⚙️ Core Technical Architecture
- **Dynamic Ingestion Cache (CRUD):** Implements targeted page-level index insertions, adjustments, and purges using metadata tags (document IDs, page numbers) and deterministic SHA-256 chunk hashes—eliminating full database rebuilds.
- **Advanced Two-Stage Retrieval Pipeline:** Combines a fast broad vector search (Approximate Nearest Neighbor via ChromaDB) for initial high recall with a high-precision local Cross-Encoder/Reranker (FlashRank) pass before forwarding chunks to the LLM.
- **Strict Bounded System Prompts:** Establishes fixed persona constraints and enforces a hard <6,000-character context parsing window limit to optimize token efficiency and prevent hallucinated out-of-scope answers.
- **Reasoning Verification Framework:** Integrates Chain-of-Thought (CoT) tracking loops that separate strategic logic into dedicated `<reasoning>...</reasoning>` tokens for audit trail validation.
- **Graceful Fault Degradation (Failover):** Utilizes structural exception handlers to intercept remote API failures, seamlessly shifting to an offline mode that displays exact retrieved local raw text chunks directly to the user.

## 🛠️ Technology Stack & Layering
1. **Document Ingestion:** PyPDF layout parsing combined with a sliding window character splitter configured for a 1,000-character ceiling and a fixed 200-character contextual overlap.
2. **Text Vectorization Engine:** Local 384-dimensional dense float vector embeddings using the open-source `all-MiniLM-L6-v2` Sentence-Transformer model.
3. **Local Vector Database Node:** Persistent database cluster utilizing ChromaDB configured with HNSW Cosine Similarity metrics.
4. **Precision Reranking:** FlashRank cross-encoder distillation down to the top 5 most definitive semantic matches.
5. **Generative Inference Gateway:** Bounded generation calls powered by the `gemini-2.5-flash` model.

## 📦 Workspace Installation & Launch

Install the core frameworks within your environment:

```bash
pip install streamlit pypdf sentence-transformers chromadb flashrank google-generativeai