"""Document store for loading and parsing Excel dictionaries and PDF manuals."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .schema import DocChunk, TermEntry

logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
    logger.warning(
        "pypdf not available. PDF loading will not work. Install with: pip install pypdf"
    )


class DocumentStore:
    """
    Abstraction layer for loading and parsing documents.

    Supports:
    - excel_dictionary: Excel files with term definitions
    - pdf_manual: PDF files with text content
    """

    def __init__(self, documents: Dict[str, Dict[str, Any]]):
        """
        Initialize the document store.

        Args:
            documents: Dictionary mapping document IDs to their metadata
        """
        self._documents = documents

    def validate_doc_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Validate that a document ID exists and return its metadata.

        Args:
            doc_id: The document ID to validate

        Returns:
            The document metadata dictionary

        Raises:
            ValueError: If the document ID is not found
        """
        if doc_id not in self._documents:
            available_ids = ", ".join(self._documents.keys())
            raise ValueError(
                f"Document '{doc_id}' not found. "
                f"Available documents: {available_ids}. "
                f"Use list_documents() to see all available documents."
            )
        return self._documents[doc_id]

    def load_term_entries(self, doc_meta: Dict[str, Any]) -> List[TermEntry]:
        """
        Load term entries from an Excel dictionary.

        Expected Excel format:
        - Columns: term, definition, synonyms (optional), related_columns (optional),
                   page (optional), extra_context (optional)
        - First row may be headers
        - Each row represents a term entry

        Args:
            doc_meta: Document metadata dictionary

        Returns:
            List of TermEntry objects

        Raises:
            ValueError: If document kind is not excel_dictionary
            RuntimeError: If loading or parsing fails
        """
        if doc_meta.get("kind") != "excel_dictionary":
            raise ValueError(
                f"load_term_entries only supports 'excel_dictionary' kind, "
                f"got '{doc_meta.get('kind')}'"
            )

        source_path = Path(doc_meta["source_path"])
        doc_id = doc_meta.get("doc_id", "unknown")

        if not source_path.exists():
            raise FileNotFoundError(
                f"Document file not found: {source_path}. "
                f"Check that the file exists and the path is correct."
            )

        try:
            logger.info(f"Loading Excel dictionary from: {source_path}")
            # Try to read Excel file
            # Support multiple sheets - read first sheet by default
            df = pd.read_excel(source_path, sheet_name=0)

            # Normalize column names (lowercase, strip whitespace)
            df.columns = df.columns.str.lower().str.strip()

            # Map common column name variations
            column_mapping = {
                "term": ["term", "terms", "word", "phrase", "name"],
                "definition": [
                    "definition",
                    "definitions",
                    "def",
                    "meaning",
                    "description",
                    "explanation",
                    "explanations",
                ],
                "synonyms": ["synonyms", "synonym", "alternatives", "aliases"],
                "related_columns": [
                    "related_columns",
                    "columns",
                    "dataset_columns",
                    "related_cols",
                ],
                "page": ["page", "pages", "pg"],
                "extra_context": ["extra_context", "context", "notes", "comments"],
            }

            # Find actual column names
            actual_columns = {}
            for target_col, possible_names in column_mapping.items():
                for possible_name in possible_names:
                    if possible_name in df.columns:
                        actual_columns[target_col] = possible_name
                        break

            # Require at least term and definition
            if "term" not in actual_columns or "definition" not in actual_columns:
                raise ValueError(
                    f"Excel file must have 'term' and 'definition' columns. "
                    f"Found columns: {list(df.columns)}"
                )

            term_entries = []
            for idx, row in df.iterrows():
                try:
                    term = str(row[actual_columns["term"]]).strip()
                    definition = str(row[actual_columns["definition"]]).strip()

                    # Skip empty rows
                    if (
                        not term
                        or not definition
                        or term == "nan"
                        or definition == "nan"
                    ):
                        continue

                    # Parse synonyms (comma-separated or list)
                    synonyms = []
                    if "synonyms" in actual_columns:
                        syn_val = row[actual_columns["synonyms"]]
                        if pd.notna(syn_val):
                            syn_str = str(syn_val).strip()
                            if syn_str:
                                # Split by comma or semicolon
                                synonyms = [
                                    s.strip()
                                    for s in syn_str.replace(";", ",").split(",")
                                    if s.strip()
                                ]

                    # Parse related columns
                    related_columns = []
                    if "related_columns" in actual_columns:
                        col_val = row[actual_columns["related_columns"]]
                        if pd.notna(col_val):
                            col_str = str(col_val).strip()
                            if col_str:
                                related_columns = [
                                    c.strip()
                                    for c in col_str.replace(";", ",").split(",")
                                    if c.strip()
                                ]

                    # Parse page number
                    page = None
                    if "page" in actual_columns:
                        page_val = row[actual_columns["page"]]
                        if pd.notna(page_val):
                            try:
                                page = int(float(page_val))
                            except (ValueError, TypeError):
                                pass

                    # Parse extra context
                    extra_context = None
                    if "extra_context" in actual_columns:
                        ctx_val = row[actual_columns["extra_context"]]
                        if pd.notna(ctx_val):
                            extra_context = str(ctx_val).strip()
                            if extra_context == "nan":
                                extra_context = None

                    term_entry = TermEntry(
                        term=term,
                        definition=definition,
                        synonyms=synonyms,
                        related_columns=related_columns,
                        source_doc_id=doc_id,
                        page=page,
                        extra_context=extra_context,
                    )
                    term_entries.append(term_entry)

                except Exception as e:
                    logger.warning(
                        f"Error parsing row {idx} in {source_path}: {e}. Skipping."
                    )
                    continue

            logger.info(f"Loaded {len(term_entries)} term entries from {source_path}")
            return term_entries

        except Exception as e:
            raise RuntimeError(
                f"Failed to load term entries from {source_path}: {str(e)}"
            ) from e

    def load_doc_chunks(
        self,
        doc_meta: Dict[str, Any],
        chunk_size: int = 2000,
        overlap: int = 300,
        min_chunk_size: int = 300,
    ) -> List[DocChunk]:
        """
        Load document chunks from a PDF manual.

        Splits PDF text into chunks for RAG-style search.

        Args:
            doc_meta: Document metadata dictionary
            chunk_size: Target size for each chunk in characters (default: 2000)
            overlap: Number of characters to overlap between chunks (default: 300)
            min_chunk_size: Minimum size for a chunk before splitting (default: 300)

        Returns:
            List of DocChunk objects

        Raises:
            ValueError: If document kind is not pdf_manual
            RuntimeError: If loading or parsing fails
        """
        if doc_meta.get("kind") != "pdf_manual":
            raise ValueError(
                f"load_doc_chunks only supports 'pdf_manual' kind, "
                f"got '{doc_meta.get('kind')}'"
            )

        if PdfReader is None:
            raise RuntimeError(
                "pypdf is not installed. Install with: pip install pypdf"
            )

        source_path = Path(doc_meta["source_path"])
        doc_id = doc_meta.get("doc_id", "unknown")

        if not source_path.exists():
            raise FileNotFoundError(
                f"Document file not found: {source_path}. "
                f"Check that the file exists and the path is correct."
            )

        try:
            logger.info(f"Loading PDF from: {source_path}")
            reader = PdfReader(str(source_path))

            chunks = []
            current_chunk_text = ""
            current_chunk_id = None
            current_chunk_start_page = None  # Track the page where the chunk started
            current_section = None

            def save_current_chunk():
                """Helper to save current chunk if it has content."""
                nonlocal \
                    current_chunk_text, \
                    current_chunk_id, \
                    current_chunk_start_page, \
                    current_section
                if (
                    current_chunk_text.strip()
                    and len(current_chunk_text.strip()) >= min_chunk_size
                ):
                    chunk_id = (
                        current_chunk_id or f"{doc_id}_chunk_{uuid.uuid4().hex[:8]}"
                    )
                    chunks.append(
                        DocChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            text=current_chunk_text.strip(),
                            page=current_chunk_start_page,
                            section_heading=current_section,
                        )
                    )
                    current_chunk_text = ""
                    current_chunk_id = None
                    current_chunk_start_page = None

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue

                    # Normalize text - replace multiple newlines with single newline
                    text = "\n".join(
                        line.strip() for line in text.split("\n") if line.strip()
                    )

                    # Split into paragraphs (double newlines) or lines
                    paragraphs = text.split("\n\n")
                    if len(paragraphs) == 1:
                        # No paragraph breaks, split by single newlines
                        paragraphs = text.split("\n")

                    for para in paragraphs:
                        para = para.strip()
                        if not para:
                            continue

                        # Try to detect section headings (more conservative)
                        # Headings are usually short, don't end with punctuation, and are standalone
                        is_heading = (
                            len(para) < 150
                            and not para.endswith((".", "。", "、", "，"))
                            and (
                                para.isupper()
                                or (
                                    len(para) > 0
                                    and para[0].isupper()
                                    and para.count(" ") < 10
                                )
                            )
                        )

                        # Set chunk start page if this is the first content in the chunk
                        if current_chunk_start_page is None:
                            current_chunk_start_page = page_num

                        if is_heading:
                            # Only break on heading if current chunk is large enough
                            if len(current_chunk_text.strip()) >= min_chunk_size:
                                save_current_chunk()

                            # Update section heading and start new chunk
                            current_section = para
                            # Add heading to chunk text (with newline)
                            current_chunk_text += f"{para}\n\n"
                            # Set the start page for this new chunk
                            if current_chunk_start_page is None:
                                current_chunk_start_page = page_num
                            continue

                        # Add paragraph to current chunk
                        para_with_newline = f"{para}\n\n"

                        # Check if adding this paragraph would exceed chunk size
                        if (
                            len(current_chunk_text) + len(para_with_newline)
                            > chunk_size
                        ):
                            # Only save if we have enough content
                            if len(current_chunk_text.strip()) >= min_chunk_size:
                                save_current_chunk()

                                # Start new chunk with overlap from previous chunk
                                if chunks and overlap > 0:
                                    prev_text = chunks[-1].text
                                    overlap_text = (
                                        prev_text[-overlap:]
                                        if len(prev_text) > overlap
                                        else prev_text
                                    )
                                    current_chunk_text = overlap_text + "\n\n"

                                # Add section heading if we have one
                                if current_section:
                                    current_chunk_text += f"{current_section}\n\n"

                                current_chunk_text += para_with_newline
                                # Set the start page for this new chunk
                                current_chunk_start_page = page_num
                            else:
                                # Current chunk is too small, keep accumulating
                                current_chunk_text += para_with_newline
                        else:
                            current_chunk_text += para_with_newline

                except Exception as e:
                    logger.warning(
                        f"Error extracting text from page {page_num} in {source_path}: {e}"
                    )
                    continue

            # Save last chunk (even if smaller than min_chunk_size)
            if current_chunk_text.strip():
                chunk_id = current_chunk_id or f"{doc_id}_chunk_{uuid.uuid4().hex[:8]}"
                chunks.append(
                    DocChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=current_chunk_text.strip(),
                        page=current_chunk_start_page,
                        section_heading=current_section,
                    )
                )

            # Post-process: merge very small chunks with adjacent chunks
            merged_chunks = []
            i = 0
            while i < len(chunks):
                current = chunks[i]

                # If chunk is too small, try to merge with next chunk
                if len(current.text) < min_chunk_size and i < len(chunks) - 1:
                    next_chunk = chunks[i + 1]
                    # Merge if combined size is reasonable
                    combined_text = f"{current.text}\n\n{next_chunk.text}"
                    if len(combined_text) <= chunk_size * 1.5:  # Allow some overflow
                        # Use the page number from the first chunk (earlier page)
                        merged_page = (
                            current.page
                            if current.page is not None
                            else next_chunk.page
                        )
                        merged_chunk = DocChunk(
                            chunk_id=current.chunk_id,
                            doc_id=current.doc_id,
                            text=combined_text.strip(),
                            page=merged_page,
                            section_heading=current.section_heading
                            or next_chunk.section_heading,
                        )
                        merged_chunks.append(merged_chunk)
                        i += 2  # Skip next chunk as it's been merged
                        continue

                merged_chunks.append(current)
                i += 1

            chunks = merged_chunks

            logger.info(
                f"Loaded {len(chunks)} chunks from {source_path} ({len(reader.pages)} pages)"
            )
            # Log chunk size statistics
            if chunks:
                chunk_sizes = [len(c.text) for c in chunks]
                logger.info(
                    f"Chunk size stats: min={min(chunk_sizes)}, max={max(chunk_sizes)}, "
                    f"avg={sum(chunk_sizes) // len(chunk_sizes)}"
                )

            return chunks

        except Exception as e:
            raise RuntimeError(
                f"Failed to load document chunks from {source_path}: {str(e)}"
            ) from e
