"""Knowledge registry for document dictionaries and manuals."""

from pathlib import Path
from typing import Any, Dict

# Calculate PROJECT_ROOT: go up 6 levels from this file
# src/mcp_server/servers/knowledge/infrastructure/knowledge_registry.py -> project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

# Registry of available knowledge documents
DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "chugai_pharma_r_d_terms": {
        "title": "Chugai Pharma R&D Terms Dictionary",
        "kind": "excel_dictionary",
        "source_path": str(PROJECT_ROOT / "data" / "chugai_pharama_r_and_d_terms.xlsx"),
        "description": "Chugai Pharmaceutical R&D terminology dictionary with terms and explanations (アンメットメディカルニーズ, 開発シナジー効果, 開発パイプライン, etc.)",
        "tags": ["pharma", "r&d", "chugai", "terms", "dictionary", "japanese"],
    },
    "medical_safety_terms": {
        "title": "Medical Safety Terms Manual",
        "kind": "pdf_manual",
        "source_path": str(PROJECT_ROOT / "data" / "medical_safety_term.pdf"),
        "description": "Medical safety terminology and definitions",
        "tags": ["safety", "medical", "terms", "pdf"],
    },
    "medical_terms": {
        "title": "Medical Terms Reference",
        "kind": "pdf_manual",
        "source_path": str(PROJECT_ROOT / "data" / "medical_terms.pdf"),
        "description": "General medical terminology reference guide",
        "tags": ["medical", "terms", "reference", "pdf"],
    },
}
