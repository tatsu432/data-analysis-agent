"""Knowledge index for RAG-style search over terms and document chunks."""

import logging
import re
from typing import List, Literal, Optional

from ..schema.output import DocChunk, KnowledgeHit, TermEntry

logger = logging.getLogger(__name__)

# Try to import embedding libraries
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. Using TF-IDF fallback. "
        "Install with: pip install sentence-transformers"
    )

# Fallback to TF-IDF
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not available. Basic text matching will be used. "
        "Install with: pip install scikit-learn"
    )


def _detect_language(text: str) -> str:
    """Detect if text is primarily Japanese or English."""
    # Simple heuristic: count Japanese characters
    japanese_chars = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", text))
    total_chars = len(re.sub(r"\s", "", text))
    if total_chars == 0:
        return "en"
    japanese_ratio = japanese_chars / total_chars
    return "ja" if japanese_ratio > 0.1 else "en"


def _normalize_query(query: str) -> str:
    """Normalize query for better matching."""
    # Remove extra whitespace
    query = re.sub(r"\s+", " ", query.strip())
    return query


class KnowledgeIndex:
    """
    In-memory index for searching over term entries and document chunks.

    Supports both embedding-based search (if sentence-transformers is available)
    and TF-IDF fallback, with hybrid search capabilities.
    """

    def __init__(
        self,
        use_embeddings: bool = True,
        embedding_model: Optional[str] = None,
        use_hybrid_search: bool = True,
    ):
        """
        Initialize the knowledge index.

        Args:
            use_embeddings: Whether to use embeddings (requires sentence-transformers)
            embedding_model: Model name for sentence-transformers. If None, auto-detects
                           based on document language. Supports multilingual models.
            use_hybrid_search: Whether to combine multiple search methods for better results
        """
        self.term_entries: List[TermEntry] = []
        self.doc_chunks: List[DocChunk] = []
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self.use_hybrid_search = use_hybrid_search
        self.embedding_model_name = embedding_model
        self.embedding_model = None

        # Will be set after documents are loaded
        self.detected_language = None

        if self.use_embeddings:
            # Will initialize model after language detection
            self.term_embeddings = None
            self.chunk_embeddings = None
        else:
            self.term_embeddings = None
            self.chunk_embeddings = None

        if SKLEARN_AVAILABLE:
            # Will configure TF-IDF after language detection
            self.vectorizer = None
            self.term_vectors = None
            self.chunk_vectors = None
            self.term_texts = []
            self.chunk_texts = []

    def _initialize_models(self) -> None:
        """Initialize embedding and TF-IDF models based on detected language."""
        if self.detected_language is None:
            # Sample text to detect language
            sample_texts = []
            for entry in self.term_entries[:10]:
                sample_texts.append(entry.term + " " + entry.definition)
            for chunk in self.doc_chunks[:10]:
                sample_texts.append(chunk.text)

            if sample_texts:
                combined_sample = " ".join(sample_texts)
                self.detected_language = _detect_language(combined_sample)
            else:
                self.detected_language = "en"  # Default to English

        logger.info(f"Detected document language: {self.detected_language}")

        # Initialize embedding model
        if self.use_embeddings:
            if self.embedding_model_name is None:
                # Auto-select model based on language
                if self.detected_language == "ja":
                    # Use multilingual model for Japanese
                    self.embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
                    logger.info(
                        "Japanese detected, using multilingual embedding model. "
                        "For better Japanese support, consider installing: "
                        "sentence-transformers with 'sentence-transformers[ja]'"
                    )
                else:
                    self.embedding_model_name = "all-MiniLM-L6-v2"

            logger.info(f"Initializing embedding model: {self.embedding_model_name}")
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to load embedding model {self.embedding_model_name}: {e}. "
                    "Falling back to TF-IDF."
                )
                self.use_embeddings = False

        # Initialize TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            # Configure based on language
            if self.detected_language == "ja":
                # For Japanese, don't use English stop words
                self.vectorizer = TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 3),  # Longer ngrams for Japanese
                    analyzer="char",  # Character-level for Japanese
                    min_df=1,
                    max_df=0.95,
                )
            else:
                self.vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                )
            logger.info("TF-IDF vectorizer configured")

    def add_term_entries(self, entries: List[TermEntry]) -> None:
        """
        Add term entries to the index.

        Args:
            entries: List of TermEntry objects to add
        """
        self.term_entries.extend(entries)
        logger.info(
            f"Added {len(entries)} term entries. Total: {len(self.term_entries)}"
        )

    def add_doc_chunks(self, chunks: List[DocChunk]) -> None:
        """
        Add document chunks to the index.

        Args:
            chunks: List of DocChunk objects to add
        """
        self.doc_chunks.extend(chunks)
        logger.info(
            f"Added {len(chunks)} document chunks. Total: {len(self.doc_chunks)}"
        )

    def build_index(self) -> None:
        """
        Build the search index from added terms and chunks.

        This should be called after all terms and chunks are added.
        """
        # Initialize models based on document language
        self._initialize_models()

        if self.use_embeddings:
            self._build_embedding_index()

        if SKLEARN_AVAILABLE:
            self._build_tfidf_index()

        if not self.use_embeddings and not SKLEARN_AVAILABLE:
            logger.info("No index building needed (using basic text matching)")

    def _build_embedding_index(self) -> None:
        """Build embedding-based index."""
        texts_to_embed = []

        # Prepare texts for term entries (include section headings in searchable text)
        term_texts = []
        for entry in self.term_entries:
            # Combine term, definition, synonyms for better search
            text_parts = [entry.term, entry.definition]
            if entry.synonyms:
                text_parts.append(" ".join(entry.synonyms))
            if entry.extra_context:
                text_parts.append(entry.extra_context)
            text = " ".join(text_parts)
            term_texts.append(text)
            texts_to_embed.append(text)

        # Prepare texts for document chunks (include section headings)
        chunk_texts = []
        for chunk in self.doc_chunks:
            # Include section heading in searchable text
            searchable_text = chunk.text
            if chunk.section_heading:
                searchable_text = f"{chunk.section_heading} {searchable_text}"
            chunk_texts.append(searchable_text)
            texts_to_embed.append(searchable_text)

        if not texts_to_embed:
            logger.warning("No texts to embed")
            return

        logger.info(f"Building embeddings for {len(texts_to_embed)} texts...")
        all_embeddings = self.embedding_model.encode(
            texts_to_embed, show_progress_bar=False, normalize_embeddings=True
        )

        # Split embeddings
        num_terms = len(term_texts)
        self.term_embeddings = all_embeddings[:num_terms]
        self.chunk_embeddings = all_embeddings[num_terms:]

        logger.info("Embedding index built successfully")

    def _build_tfidf_index(self) -> None:
        """Build TF-IDF based index."""
        # Prepare texts for term entries
        self.term_texts = []
        for entry in self.term_entries:
            text_parts = [entry.term, entry.definition]
            if entry.synonyms:
                text_parts.append(" ".join(entry.synonyms))
            if entry.extra_context:
                text_parts.append(entry.extra_context)
            text = " ".join(text_parts)
            self.term_texts.append(text)

        # Prepare texts for document chunks (include section headings)
        self.chunk_texts = []
        for chunk in self.doc_chunks:
            searchable_text = chunk.text
            if chunk.section_heading:
                # Weight section heading more by repeating it
                searchable_text = (
                    f"{chunk.section_heading} {chunk.section_heading} {searchable_text}"
                )
            self.chunk_texts.append(searchable_text)

        all_texts = self.term_texts + self.chunk_texts
        if not all_texts:
            logger.warning("No texts to vectorize")
            return

        logger.info(f"Building TF-IDF vectors for {len(all_texts)} texts...")
        all_vectors = self.vectorizer.fit_transform(all_texts)

        # Split vectors
        num_terms = len(self.term_texts)
        self.term_vectors = all_vectors[:num_terms]
        self.chunk_vectors = all_vectors[num_terms:]

        logger.info("TF-IDF index built successfully")

    def _compute_exact_match_score(self, query: str, text: str) -> float:
        """Compute exact match score (boost for exact matches)."""
        query_lower = query.lower().strip()
        text_lower = text.lower()

        # Exact phrase match
        if query_lower in text_lower:
            # Boost if match is at the beginning
            position = text_lower.find(query_lower)
            if position == 0:
                return 2.0  # Strong boost for beginning match
            elif position < 50:
                return 1.5  # Moderate boost for early match
            return 1.2  # Small boost for any match

        # Word-level matches
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        common_words = query_words & text_words

        if len(query_words) > 0:
            word_match_ratio = len(common_words) / len(query_words)
            return word_match_ratio * 0.5  # Smaller boost for word matches

        return 0.0

    def search(
        self,
        query: str,
        scopes: List[Literal["terms", "docs"]] = None,
        top_k: int = 5,
    ) -> List[KnowledgeHit]:
        """
        Search the knowledge base with hybrid search.

        Args:
            query: Search query string
            scopes: List of scopes to search ("terms", "docs", or both)
            top_k: Number of results to return

        Returns:
            List of KnowledgeHit objects, sorted by score (descending)
        """
        if scopes is None:
            scopes = ["terms", "docs"]

        query = _normalize_query(query)
        results = []

        if "terms" in scopes and self.term_entries:
            term_results = self._search_terms(query, top_k * 2)  # Get more candidates
            results.extend(term_results)

        if "docs" in scopes and self.doc_chunks:
            doc_results = self._search_chunks(query, top_k * 2)  # Get more candidates
            results.extend(doc_results)

        # Rerank results with hybrid scoring
        if self.use_hybrid_search:
            results = self._rerank_results(query, results)

        # Sort by final score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        # Return top_k overall results
        return results[:top_k]

    def _rerank_results(
        self, query: str, results: List[KnowledgeHit]
    ) -> List[KnowledgeHit]:
        """Rerank results using hybrid scoring (combines multiple signals)."""
        reranked = []

        for hit in results:
            base_score = hit.score
            exact_match_boost = 0.0

            # Get text to check for exact matches
            if hit.kind == "term" and hit.term_entry:
                text = f"{hit.term_entry.term} {hit.term_entry.definition}"
                # Boost exact matches in term name
                if query.lower() in hit.term_entry.term.lower():
                    exact_match_boost += 1.0
            elif hit.kind == "chunk" and hit.chunk:
                text = hit.chunk.text
                if hit.chunk.section_heading:
                    # Boost if query matches section heading
                    if query.lower() in hit.chunk.section_heading.lower():
                        exact_match_boost += 0.8

            # Compute exact match score
            exact_match_score = self._compute_exact_match_score(query, text)

            # Combine scores (weighted combination)
            final_score = (
                (base_score * 0.7) + (exact_match_score * 0.3) + exact_match_boost
            )

            # Create new hit with reranked score
            reranked_hit = KnowledgeHit(
                kind=hit.kind,
                score=final_score,
                term_entry=hit.term_entry,
                chunk=hit.chunk,
            )
            reranked.append(reranked_hit)

        return reranked

    def _search_terms(self, query: str, top_k: int) -> List[KnowledgeHit]:
        """Search term entries with hybrid approach."""
        if not self.term_entries:
            return []

        scores = {}

        # Method 1: Embedding-based search
        if self.use_embeddings and self.term_embeddings is not None:
            query_embedding = self.embedding_model.encode(
                [query], show_progress_bar=False, normalize_embeddings=True
            )[0]

            # Vectorized similarity computation
            similarities = np.dot(self.term_embeddings, query_embedding)

            for i, sim_score in enumerate(similarities):
                scores[i] = scores.get(i, 0.0) + float(sim_score) * 0.6

        # Method 2: TF-IDF search
        if SKLEARN_AVAILABLE and self.term_vectors is not None:
            query_vector = self.vectorizer.transform([query])
            similarities_matrix = cosine_similarity(query_vector, self.term_vectors)

            for i, sim_score in enumerate(similarities_matrix[0]):
                scores[i] = scores.get(i, 0.0) + float(sim_score) * 0.4

        # Method 3: Exact match boost (always applied)
        for i, entry in enumerate(self.term_entries):
            exact_score = self._compute_exact_match_score(query, entry.term)
            if exact_score > 0:
                scores[i] = scores.get(i, 0.0) + exact_score * 0.3

        # Convert to list and sort
        similarities = [(i, score) for i, score in scores.items() if score > 0]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_k]

        hits = []
        for idx, score in top_similarities:
            entry = self.term_entries[idx]
            hits.append(
                KnowledgeHit(
                    kind="term",
                    score=score,
                    term_entry=entry,
                    chunk=None,
                )
            )

        return hits

    def _search_chunks(self, query: str, top_k: int) -> List[KnowledgeHit]:
        """Search document chunks with hybrid approach."""
        if not self.doc_chunks:
            return []

        scores = {}

        # Method 1: Embedding-based search
        if self.use_embeddings and self.chunk_embeddings is not None:
            query_embedding = self.embedding_model.encode(
                [query], show_progress_bar=False, normalize_embeddings=True
            )[0]

            # Vectorized similarity computation
            similarities = np.dot(self.chunk_embeddings, query_embedding)

            for i, sim_score in enumerate(similarities):
                scores[i] = scores.get(i, 0.0) + float(sim_score) * 0.6

        # Method 2: TF-IDF search
        if SKLEARN_AVAILABLE and self.chunk_vectors is not None:
            query_vector = self.vectorizer.transform([query])
            similarities_matrix = cosine_similarity(query_vector, self.chunk_vectors)

            for i, sim_score in enumerate(similarities_matrix[0]):
                scores[i] = scores.get(i, 0.0) + float(sim_score) * 0.4

        # Method 3: Exact match boost (always applied)
        query_lower = query.lower()
        for i, chunk in enumerate(self.doc_chunks):
            # Check chunk text
            exact_score = self._compute_exact_match_score(query, chunk.text)
            # Boost if section heading matches
            if chunk.section_heading and query_lower in chunk.section_heading.lower():
                exact_score += 0.5

            if exact_score > 0:
                scores[i] = scores.get(i, 0.0) + exact_score * 0.3

        # Convert to list and sort
        similarities = [(i, score) for i, score in scores.items() if score > 0]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_k]

        hits = []
        for idx, score in top_similarities:
            chunk = self.doc_chunks[idx]
            hits.append(
                KnowledgeHit(
                    kind="chunk",
                    score=score,
                    term_entry=None,
                    chunk=chunk,
                )
            )

        return hits


# Global knowledge index instance
_global_knowledge_index: Optional[KnowledgeIndex] = None


def build_global_knowledge_index(
    documents: dict,
    document_store,
    use_embeddings: bool = True,
    embedding_model: Optional[str] = None,
    use_hybrid_search: bool = True,
) -> KnowledgeIndex:
    """
    Build a global knowledge index from all documents in the registry.

    Args:
        documents: Dictionary of document metadata (from knowledge_registry)
        document_store: DocumentStore instance
        use_embeddings: Whether to use embeddings
        embedding_model: Optional embedding model name (auto-detected if None)
        use_hybrid_search: Whether to use hybrid search combining multiple methods

    Returns:
        KnowledgeIndex instance
    """
    global _global_knowledge_index

    logger.info("Building global knowledge index...")
    index = KnowledgeIndex(
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
        use_hybrid_search=use_hybrid_search,
    )

    # Load all documents
    for doc_id, doc_meta in documents.items():
        doc_meta_with_id = {**doc_meta, "doc_id": doc_id}
        try:
            if doc_meta["kind"] == "excel_dictionary":
                entries = document_store.load_term_entries(doc_meta_with_id)
                index.add_term_entries(entries)
            elif doc_meta["kind"] == "pdf_manual":
                chunks = document_store.load_doc_chunks(doc_meta_with_id)
                index.add_doc_chunks(chunks)
        except Exception as e:
            logger.error(f"Failed to load document '{doc_id}': {e}")
            continue

    # Build the index
    index.build_index()

    logger.info(
        f"Knowledge index built: {len(index.term_entries)} terms, "
        f"{len(index.doc_chunks)} chunks"
    )

    _global_knowledge_index = index
    return index


def get_global_knowledge_index() -> Optional[KnowledgeIndex]:
    """Get the global knowledge index instance."""
    return _global_knowledge_index
