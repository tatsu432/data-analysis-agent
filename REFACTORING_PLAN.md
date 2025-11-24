# MCP Server Refactoring Plan

## Goal
Refactor the MCP server code to follow clean architecture pattern similar to `anyai-mcp-server`, with clear separation of concerns:
- **server.py**: Thin FastMCP layer that delegates to use cases
- **usecases/**: Business logic organized by use case
- **infrastructure/**: External dependencies (repositories, API clients, etc.)
- **schema/**: Input/output schemas (Pydantic models)
- **const.py**: Domain constants

## Current Structure
```
src/mcp_server/
├── analysis_tools.py (858 lines - needs refactoring)
├── knowledge_tools.py (needs refactoring)
├── schema.py (mixed schemas - needs splitting)
├── dataset_store.py (infrastructure)
├── datasets_registry.py (infrastructure)
├── document_store.py (infrastructure)
├── knowledge_index.py (infrastructure)
├── knowledge_registry.py (infrastructure)
└── servers/
    ├── analysis/ (partially created)
    ├── knowledge/ (needs creation)
    └── confluence/ (needs refactoring)
```

## Target Structure
```
src/mcp_server/
├── server.py (main unified server)
└── servers/
    ├── analysis/
    │   ├── server.py (FastMCP tools)
    │   ├── schema/
    │   │   ├── input.py
    │   │   └── output.py
    │   ├── usecases/
    │   │   ├── list_datasets_usecase.py
    │   │   ├── get_dataset_schema_usecase.py
    │   │   └── run_analysis_usecase.py
    │   ├── infrastructure/
    │   │   ├── dataset_store.py
    │   │   └── datasets_registry.py
    │   └── const.py
    ├── knowledge/
    │   ├── server.py
    │   ├── schema/
    │   │   ├── input.py
    │   │   └── output.py
    │   ├── usecases/
    │   │   ├── list_documents_usecase.py
    │   │   ├── get_term_definition_usecase.py
    │   │   └── search_knowledge_usecase.py
    │   ├── infrastructure/
    │   │   ├── document_store.py
    │   │   ├── knowledge_index.py
    │   │   └── knowledge_registry.py
    │   └── const.py
    └── confluence/
        ├── server.py (refactor from tools.py)
        ├── schema/
        │   ├── input.py
        │   └── output.py
        ├── usecases/
        │   ├── search_pages_usecase.py
        │   ├── get_page_usecase.py
        │   ├── create_page_usecase.py
        │   └── update_page_usecase.py
        ├── infrastructure/
        │   └── confluence_client.py
        └── const.py
```

## Implementation Steps

### Phase 1: Analysis Domain ✅ (In Progress)
- [x] Create schema/input.py and schema/output.py
- [x] Create infrastructure/ (copy and update imports)
- [x] Create usecases/list_datasets_usecase.py
- [x] Create usecases/get_dataset_schema_usecase.py
- [ ] Create usecases/run_analysis_usecase.py (complex - needs helper functions)
- [ ] Create server.py (thin FastMCP layer)
- [ ] Update imports in infrastructure files

### Phase 2: Knowledge Domain
- [ ] Create schema/input.py and schema/output.py
- [ ] Create infrastructure/ (copy and update imports)
- [ ] Create usecases/
- [ ] Create server.py

### Phase 3: Confluence Domain
- [ ] Refactor tools.py → server.py
- [ ] Create schema/
- [ ] Create usecases/
- [ ] Create infrastructure/confluence_client.py

### Phase 4: Integration
- [ ] Update main server.py to import from new structure
- [ ] Remove old files (analysis_tools.py, knowledge_tools.py, etc.)
- [ ] Update all imports
- [ ] Test all functionality

## Notes
- Helper functions from analysis_tools.py (like `_preprocess_code_for_deprecated_styles`, `_validate_plot`) should be moved to infrastructure/utils.py or kept as private methods in use cases
- The run_analysis_usecase is complex and may need to import helper functions from the old analysis_tools.py initially, then gradually refactor
- All domain-specific constants should move to const.py files

