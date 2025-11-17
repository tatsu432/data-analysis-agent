# Data Analysis Agent Prototype

This project is a prototype demonstrating an LLM-powered data analysis agent that can answer analytical questions by generating Python code, executing it, analyzing the results, and responding in natural language.

The goal is to show how an agent can perform end-to-end analysis on internal datasets (such as S3 or Redshift) in the future. For the prototype, the agent operates on multiple local CSV files including COVID-19 patient counts for Japanese prefectures and patient data.

---

## What This Prototype Demonstrates

The agent can:

1. **Answer terminology and document questions**
   - Look up definitions from Excel dictionaries and PDF manuals
   - Search knowledge base using hybrid search (embeddings + TF-IDF)
   - Answer pure document/terminology questions without data analysis
   - Example: "What does GP mean?" or "開発シナジー効果とは？"

2. **Interpret natural-language analytical questions**  
   Example: "How does the number of patients vary from January to July 2025 in Tokyo?"

3. **Enrich data analysis with domain knowledge**
   - Automatically identify domain-specific terms in queries
   - Look up term definitions and map them to dataset columns
   - Use knowledge context to write more accurate analysis code
   - Example: Understands "GP" maps to `channel_type == "GP"` in datasets

4. Plan the required analytical steps (filtering, grouping, aggregating, plotting)

5. Generate executable Python code using pandas and matplotlib

6. Execute the generated code through a controlled tool interface

7. Capture results  
   - dataframe preview  
   - stdout  
   - generated plot (PNG)

8. Understand and summarize the results in natural language, explaining domain terms when relevant

9. **Export analysis results to Confluence**
   - Create structured Confluence reports from analysis results
   - Automatically format content with sections, code snippets, and results
   - Store page information for future reference

10. **Read and summarize existing Confluence pages**
    - Search for relevant Confluence pages
    - Understand and handle different query types (meta-questions, specific searches)
    - Summarize or answer questions based on Confluence content

This provides a clear vertical slice of an "LLM analyst" workflow with integrated knowledge management, Confluence documentation, and is designed to be extensible to enterprise data.

---

## Architecture Overview

High-Level Components:

- **LangGraph Server** (`src/langgraph_server/`)  
  - LangGraph agent that interprets queries  
  - writes Python code  
  - retries on errors  
  - summarizes results  
  - loads tools from multiple MCP servers via `langchain-mcp-adapters`
  - Includes query classification to route between document QA, data analysis, and Confluence operations
  - Supports knowledge enrichment for domain-specific terms
  - Integrates Confluence read/write workflows

- **Data Analysis MCP Server** (`src/mcp_server/`)  
  - FastMCP server exposing data analysis tools
  - Dataset Registry (`datasets_registry.py`): Centralized registry of available datasets with metadata
  - Dataset Store (`dataset_store.py`): Abstraction layer for loading datasets from various storage backends
    - Supports `local_csv`: Local CSV files
    - Supports `s3_csv`: CSV files stored on S3 (using s3fs or boto3)
  - Execution Tool (`run_analysis` in `analysis_tools.py`)  
    - loads one or more datasets into dataframes via DatasetStore
    - executes generated Python code safely  
    - returns previews and plots  
    - supports multi-dataset analysis
    - includes plot validation to detect empty/invalid plots
    - supports advanced libraries: sklearn, statsmodels, torch, Prophet, pmdarima, arch
  - Dataset Tools:
    - `list_datasets`: Lists all available datasets with their IDs, descriptions, code aliases, and storage locations
    - `get_dataset_schema(dataset_id)`: Returns column names, datatypes, sample rows, and row count for a specific dataset
  - Knowledge Layer:
    - Knowledge Registry (`knowledge_registry.py`): Centralized registry of available knowledge documents (Excel dictionaries and PDF manuals)
    - Document Store (`document_store.py`): Loads and parses Excel dictionaries and PDF manuals
      - Excel dictionaries: Parses term definitions with flexible column mapping
      - PDF manuals: Extracts text and chunks documents for RAG-style search
    - Knowledge Index (`knowledge_index.py`): In-memory vector search using embeddings and TF-IDF
      - Auto-detects document language (Japanese/English)
      - Uses multilingual embeddings for Japanese documents
      - Hybrid search combining embeddings, TF-IDF, and exact matching
    - Knowledge Tools:
      - `list_documents`: Lists available knowledge documents with metadata
      - `get_document_metadata(doc_id)`: Gets metadata for a specific document
      - `get_term_definition(term)`: Gets definition of a specific term (exact match + similarity search)
      - `search_knowledge(query, scopes, top_k)`: Searches the knowledge base using hybrid search
  - Utilities (`utils.py`): Shared utility functions including automatic datetime column detection and conversion  

- **Confluence MCP Server** (`confluence_mcp_server/`)  
  - FastMCP server exposing Confluence integration tools
  - Provides tools for searching, reading, creating, and updating Confluence pages
  - Handles markdown-to-HTML conversion and format compatibility
  - Filters search results to match Confluence Content view behavior

The architecture follows a separation of concerns pattern where:
- MCP tools are defined in separate servers (`mcp_server/` and `confluence_mcp_server/`) and exposed via FastMCP
- LangGraph agent flow is defined in `langgraph_server/` and consumes tools from multiple MCP servers
- Each MCP server can be developed, deployed, and scaled independently

### Agent Workflow

The agent workflow includes query classification and routing:

![Agent Workflow](agent_workflow.png)

The workflow routes queries through:
1. **Classification**: 
   - Determines if query is DOCUMENT_QA, DATA_ANALYSIS, or BOTH
   - Detects Confluence-related intents (`FROM_ANALYSIS` for export, `FROM_CONFLUENCE` for reading)
2. **Document QA Path**: For pure terminology questions, uses knowledge tools to answer
3. **Knowledge Enrichment**: For queries needing both knowledge and analysis, enriches with domain term definitions
4. **Data Analysis Path**: For data analysis queries, proceeds directly to code generation and execution
5. **Confluence Export Path**: After successful analysis, if user requested export, creates Confluence page with results
6. **Confluence Read Path**: For Confluence queries, searches pages, selects relevant content, and summarizes

---

## Repository Structure
```
project-root/
│
├── data/
│ ├── newly_confirmed_cases_daily.csv
│ ├── jpm_patient_data.csv
│ ├── jamdas_patient_data.csv
│ ├── mr_activity_data.csv
│ ├── chugai_pharama_r_and_d_terms.xlsx # Knowledge dictionary (Excel)
│ ├── medical_safety_term.pdf # Knowledge manual (PDF)
│ ├── medical_terms.pdf # Knowledge manual (PDF)
│ └── kaggle_covid/ # Additional COVID-19 datasets (optional)
│
├── img/ # Generated plot images are saved here
│
├── src/
│ ├── langgraph_server/
│ │ ├── graph.py # LangGraph agent definition
│ │ ├── prompts.py # Agent prompts and reasoning steps
│ │ ├── mcp_tool_loader.py # Loads tools from MCP server(s)
│ │ ├── settings.py # LangGraph server settings
│ │ ├── __main__.py # Entry point for CLI mode
│ │ └── generate_diagram.py # Script to generate workflow diagram
│ │
│ ├── mcp_server/
│ │ ├── server.py # FastMCP server entry point
│ │ ├── analysis_tools.py # MCP tools (list_datasets, get_dataset_schema, run_analysis)
│ │ ├── datasets_registry.py # Dataset registry with metadata
│ │ ├── dataset_store.py # Abstraction layer for loading datasets (local_csv, s3_csv)
│ │ ├── knowledge_registry.py # Knowledge document registry with metadata
│ │ ├── document_store.py # Loads and parses Excel dictionaries and PDF manuals
│ │ ├── knowledge_index.py # In-memory vector search (embeddings + TF-IDF)
│ │ ├── knowledge_tools.py # MCP tools (list_documents, get_term_definition, search_knowledge)
│ │ ├── schema.py # Pydantic schemas for tool inputs/outputs
│ │ ├── settings.py # MCP server settings
│ │ ├── utils.py # Shared utility functions (datetime conversion, etc.)
│ │ └── __main__.py # Entry point for running the MCP server
│ │
│ ├── app/
│ │ └── ui.py # Streamlit UI
│
├── confluence_mcp_server/
│ ├── server.py # Confluence MCP server entry point
│ ├── confluence_tools.py # Confluence MCP tools (search, get, create, update pages)
│ ├── __init__.py
│ └── README.md # Quick start guide for Confluence MCP server
│
└── README.md
```

---

## Tools

### `list_datasets()`

Returns a list of all available datasets with:
- Dataset IDs
- Descriptions
- Code aliases (variable names for use in code)
- Storage kind (`local_csv` or `s3_csv`)
- Location hint (file path for local, S3 URI for S3)

Use this to discover which datasets are available for analysis.

---

### `get_dataset_schema(dataset_id: str)`

Returns schema information for a specific dataset:
- Column names  
- Inferred dtypes  
- First 5 rows (JSON)
- Row count
- Dataset description

Helps the agent avoid hallucinating nonexistent columns or datatypes.

**Example**: `get_dataset_schema("covid_new_cases_daily")` or `get_dataset_schema("jpm_patient_data")`

---

### `run_analysis(code: str, dataset_ids: list[str], primary_dataset_id: str | None = None)`

Executes generated Python code in a controlled environment with one or more datasets.

**Parameters:**
- `code`: Python code string to execute
- `dataset_ids`: List of dataset IDs to load (e.g., `["covid_new_cases_daily"]` or `["jpm_patient_data", "covid_new_cases_daily"]`)
- `primary_dataset_id`: Optional primary dataset ID (if provided, this dataset is available as `df`)

**Dataset Access in Code:**
- Primary dataset: If `primary_dataset_id` is specified, available as `df`
- Single dataset: If only one dataset is loaded, automatically available as `df`
- All datasets: Access via `dfs[dataset_id]` dictionary
- Code aliases: Each dataset has a code_name (e.g., `df_covid_daily`, `df_jpm_patients`)

**Execution rules:**
- Allowed libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), sklearn, statsmodels, torch, Prophet, pmdarima, arch
  - sklearn modules: linear_model, metrics, model_selection, preprocessing
  - statsmodels: Comprehensive statistical modeling (OLS, ARIMA, SARIMAX, VAR, etc.)
  - PyTorch: Deep learning and tensor operations
  - Time series: Prophet (Facebook Prophet), pmdarima (Auto ARIMA), arch (ARCH/GARCH)
- Date columns are automatically converted to datetime
- Plot validation: Automatically validates plots to detect empty/invalid plots
- Captures:
  - stdout  
  - error  
  - preview of `result_df` if defined  
  - saved plot with timestamped filename (e.g., `plot_20251115_212901.png`) if created
  - plot validation status and messages

**Example for single dataset:**
```python
run_analysis(
    code="result_df = df.head(10)",
    dataset_ids=["covid_new_cases_daily"],
    primary_dataset_id="covid_new_cases_daily"
)
```

**Example for multiple datasets:**
```python
run_analysis(
    code="merged = df_jpm_patients.merge(df_covid_daily, on='date', how='left'); result_df = merged.head(10)",
    dataset_ids=["jpm_patient_data", "covid_new_cases_daily"],
    primary_dataset_id="jpm_patient_data"
)
```

---

## Agent Behavior

The agent uses intelligent query classification and routing:

### Query Classification

Every query is first classified into one of three categories:
- **DOCUMENT_QA**: Pure terminology/definition questions (e.g., "What does GP mean?")
- **DATA_ANALYSIS**: Data analysis questions with no ambiguous domain terms (e.g., "Show me COVID cases in Tokyo")
- **BOTH**: Questions needing both knowledge lookup and data analysis (e.g., "Compare GP vs HP patient counts")

### Workflow Paths

**Document QA Path:**
1. Query classified as DOCUMENT_QA
2. Agent uses knowledge tools (`get_term_definition`, `search_knowledge`)
3. Returns answer based on knowledge base
4. No data analysis performed

**Data Analysis Path:**
1. Query classified as DATA_ANALYSIS
2. Agent proceeds directly to:
   - Identify datasets needed
   - Plan analytical steps
   - Generate Python code
   - Execute code via `run_analysis`
   - Validate results
   - Summarize findings

**Hybrid Path (BOTH):**
1. Query classified as BOTH
2. Knowledge enrichment step:
   - Identifies domain-specific terms
   - Looks up term definitions
   - Builds knowledge context mapping terms to dataset columns
3. Agent uses knowledge context to:
   - Map domain terms to columns (e.g., "GP" → `channel_type == "GP"`)
   - Generate more accurate analysis code
   - Explain terms in natural language response
4. Proceeds with data analysis as above

### Error Handling

If execution errors occur:
- The agent analyzes the traceback
- Regenerates corrected code
- Retries once

---

## Knowledge Tools

### `list_documents()`

Returns a list of all available knowledge documents (Excel dictionaries and PDF manuals) with:
- Document IDs
- Titles
- Document kind (`excel_dictionary` or `pdf_manual`)
- Source paths
- Descriptions
- Tags

Use this to discover which knowledge documents are available.

---

### `get_document_metadata(doc_id: str)`

Returns metadata for a specific knowledge document.

**Example**: `get_document_metadata("chugai_pharma_r_d_terms")`

---

### `get_term_definition(term: str)`

Gets the definition of a specific term from the knowledge base.

**Behavior:**
- First tries exact match (case-insensitive)
- Checks synonyms if no exact match
- Falls back to similarity search if needed
- Returns `TermEntry` object with definition, synonyms, related columns, etc.

**Example**: `get_term_definition("GP")` or `get_term_definition("開発シナジー効果")`

---

### `search_knowledge(query: str, scopes: Optional[List[str]] = None, top_k: int = 5)`

Searches the knowledge base for relevant information using hybrid search.

**Parameters:**
- `query`: Search query string
- `scopes`: List of scopes to search - `["terms"]` for term definitions only, `["docs"]` for document chunks only, or `["terms", "docs"]` for both (default: both)
- `top_k`: Maximum number of results to return (default: 5)

**Search Methods:**
- **Embedding-based search**: Uses sentence-transformers (multilingual for Japanese)
- **TF-IDF search**: Character-level for Japanese, word-level for English
- **Exact match boosting**: Prioritizes exact phrase matches
- **Hybrid scoring**: Combines multiple signals for better relevance

**Returns:**
- List of `KnowledgeHit` objects with:
  - `kind`: "term" or "chunk"
  - `score`: Similarity score
  - `term_entry`: TermEntry if kind is "term"
  - `chunk`: DocChunk if kind is "chunk" (includes page numbers and section headings)

**Example**: `search_knowledge("pharmaceutical R&D terms", scopes=["terms"], top_k=3)`

---

## Example Queries

You can ask:

- How does the number of patients vary from January to July 2022 in Tokyo?
- 2022年1月から2022年12月までの東京のコロナウイルス感染者数を図にして、要約して
- Generate and compare the line plots of the number of patients from January to August 2022 in Tokyo, Chiba, Saitama, Kanagawa.
- What characteristics does the patient count data have overall?
- Can you model the Tokyo's covid case and tell me the model clearly?
- Can you compare the each product's number of patients over the time for GP only?
- HPのみに絞った上で、ラゲブリオ、パケロビッド、ゾコーバのそれぞれのコロナウイルス治療患者数を2022年1月から2024年12月までで図にして
- Can you generate the line plots of the number of the patients for each product only for those at risk over the time?
- Can you create a regression model where we predict the number of patient for LAGEVRIO by the MR activities? Tell me the fitted model and MAPE.
- MRの活動からラゲブリオの患者人数を予測する回帰モデルを作成して、予測精度についてまとめて、回帰モデルをわかりやすく説明して
- Generate the line plots of the number of those who are recovered from COVID over the time for South Korea, Canada, France, and US.
- 韓国、カナダ、フランス、アメリカに関して、コロナウイルス治癒患者数を時系列でプロットして
- 世界規模の動向を知るために、韓国、カナダ、フランス、アメリカに関して、コロナウイルス治癒患者数を時系列でプロットして
- 患者経験調査とは何？
- What does Hp mean?
- 開発シナジー効果とは？
- What kind of confluence pages can I see?
- どのようなConfluenceのページがある？
- Summarize the result and create a confluence page about your analysis.
- Create a Confluence report from this analysis.
- コロナウイルスに関する今までの分析をまとめて、Confluenceのページに投稿して
---



## Available Datasets

The system currently supports the following datasets:

1. **jpm_patient_data**: Patient data by product (LAGEVRIO, PAXLOVID, XOCOVA) with HP/GP breakdown
2. **jamdas_patient_data**: Patient data with at-risk and DDI prescription information (GP only)
3. **covid_new_cases_daily**: COVID-19 newly confirmed cases daily data for Japanese prefectures (local CSV)
4. **mr_activity_data**: MR activity data by prefecture, month, and HP/GP type (detailing visits, emails, seminars) for all 47 prefectures from 2023-04 to 2025-09
5. **covid_full_grouped**: Global COVID-19 data grouped by country/region and date (S3-based)

Datasets can be stored locally (`local_csv`) or on S3 (`s3_csv`). The DatasetStore abstraction handles loading from both sources transparently.

---

## Available Knowledge Documents

The system currently supports the following knowledge documents:

1. **chugai_pharma_r_d_terms** (Excel Dictionary)
   - Chugai Pharmaceutical R&D terminology dictionary
   - Contains terms and explanations in Japanese (e.g., アンメットメディカルニーズ, 開発シナジー効果, 開発パイプライン)
   - Format: Excel with "Term" and "Explanation" columns

2. **medical_safety_terms** (PDF Manual)
   - Medical safety terminology and definitions
   - Extracted and chunked for RAG-style search

3. **medical_terms** (PDF Manual)
   - General medical terminology reference guide
   - Extracted and chunked for RAG-style search

Knowledge documents are automatically indexed on MCP server startup using:
- **Multilingual embeddings** for Japanese documents (paraphrase-multilingual-MiniLM-L12-v2)
- **TF-IDF** with language-appropriate settings (character-level for Japanese, word-level for English)
- **Hybrid search** combining embeddings, TF-IDF, and exact matching for optimal results

---

## Technical Details

### Knowledge Index Architecture

The knowledge index uses an **in-memory vector search** approach:

- **No database required**: All embeddings and vectors stored in memory as NumPy arrays
- **Fast search**: Vectorized operations using NumPy for efficient similarity computation
- **Hybrid search**: Combines multiple search methods:
  - Embedding-based similarity (60% weight)
  - TF-IDF similarity (40% weight)
  - Exact match boosting (30% additional weight)
- **Language-aware**: Auto-detects document language and uses appropriate models
  - Japanese: Multilingual embeddings + character-level TF-IDF
  - English: English embeddings + word-level TF-IDF
- **Memory efficient**: Suitable for small-to-medium document collections (thousands of documents)

### Document Processing

- **Excel dictionaries**: Flexible column mapping supports various formats (Term/Definition, Term/Explanation, etc.)
- **PDF manuals**: Intelligent chunking with:
  - Target chunk size: 2000 characters
  - Minimum chunk size: 300 characters
  - Overlap: 300 characters for context preservation
  - Section heading detection and weighting
  - Automatic merging of small chunks

---

## Confluence Integration via MCP

The agent supports integration with Confluence for persistent documentation of analyses and retrieval of previous analyses. This integration uses MCP (Model Context Protocol) tools, allowing the agent to read from and write to Confluence pages.

### Features

1. **Export Analysis to Confluence**
   - After running a data analysis, users can request to create a Confluence report
   - The agent automatically:
     - Extracts analysis context (question, datasets used, code, results, plots, summary)
     - Generates a well-structured Confluence page draft using an LLM with sections:
       - Overview / Business Question
       - Datasets Used
       - Methodology (with code snippets in markdown)
       - Results (tables and plot references)
       - Interpretation / Caveats
       - Reproduction Steps
     - Converts markdown to HTML for Confluence compatibility
     - Creates the page in the configured Confluence space (default: `CONFLUENCE_SPACE_KEY_ANALYTICS`)
     - Returns the page ID and URL to the user
     - Stores page information in state for future reference

2. **Read and Summarize Existing Confluence Pages**
   - Users can ask questions about existing Confluence content
   - The agent intelligently handles different query types:
     - **Meta-questions** (e.g., "What kind of pages can I see?"): Shows a list of available pages
     - **Specific searches** (e.g., "GP vs HP analysis"): Searches and selects the most relevant page
   - The agent automatically:
     - Understands and reformulates the user's query for effective Confluence search
     - Searches Confluence pages by title (matching Content view behavior)
     - Filters out attachments, archived pages, and invalid entries
     - Selects the most relevant page using LLM-based ranking
     - Fetches the full page content
     - Summarizes or answers questions based on the content

### Configuration

To enable Confluence integration:

1. **Set up a Confluence MCP Server**
   
   **Quick Start (Recommended):** A ready-to-use Confluence MCP server is included in the `confluence_mcp_server/` directory.
   
   ```bash
   # Install dependencies (if not already installed)
   pip install atlassian-python-api markdown
   
   # Add to your .env file (in the project root):
   CONFLUENCE_URL=https://yourcompany.atlassian.net
   CONFLUENCE_USERNAME=your.email@company.com
   CONFLUENCE_API_TOKEN=your_api_token_here
   
   # Run the server
   python confluence_mcp_server/server.py
   ```
   
   The server will start on port 8083 by default and expose Confluence tools via MCP.
   
   **For detailed setup instructions**, see [CONFLUENCE_MCP_SETUP.md](CONFLUENCE_MCP_SETUP.md) which covers:
   - Step-by-step setup guide
   - Alternative options (Atlassian official, CData)
   - Troubleshooting tips
   - Getting API tokens

2. **Configure Environment Variables in your main project**
   ```bash
   CONFLUENCE_MCP_SERVER_URL=http://localhost:8083/mcp  # URL of your Confluence MCP server
   CONFLUENCE_SPACE_KEY_ANALYTICS=ANALYTICS  # Default space key for analytics reports
   ```

3. **Start the Confluence MCP Server**
   - Run the server: `python confluence_mcp_server/server.py`
   - Ensure it's running and accessible at the configured URL
   - The server uses your Confluence API token for authentication

### Usage Examples

**Export to Confluence:**
```
User: "Create a Confluence report from this analysis."
User: "Write this up as a Confluence page under the Merck analytics space."
User: "Document these results in Confluence for the product owner."
```

**Read from Confluence:**
```
User: "What kind of confluence pages can I see?"
User: "What were the main takeaways from the last GP vs HP share analysis in Confluence?"
User: "Summarize the latest LAGEVRIO forecasting report from Confluence."
User: "Find our earlier analysis on MR activity and patient counts and tell me the result."
```

**Note**: The agent intelligently handles meta-questions (like "What kind of pages can I see?") by showing a list of available pages, while specific queries trigger a search and summarization workflow.

### Architecture Notes

- The Confluence integration is **additive** - existing functionality remains unchanged
- Confluence tools are loaded from a separate MCP server alongside the data analysis tools
- The agent automatically routes to Confluence subflows based on user intent (detected via `doc_action` classification)
- The integration includes:
  - Query understanding and reformulation for better search results
  - Intelligent filtering to match Confluence Content view behavior
  - Automatic markdown-to-HTML conversion for page creation
  - Comprehensive error handling with user-friendly messages
- If the Confluence MCP server is not configured, the agent continues to work normally without Confluence features
- The agent supports multiple MCP servers simultaneously (data analysis + Confluence)

### Confluence Tools

The agent uses the following Confluence MCP tools (provided by the Confluence MCP server):

1. **`confluence_search_pages(query, space_key=None, limit=10)`**
   - Searches for Confluence pages by title (matches Content view behavior)
   - Filters out attachments, archived pages, and invalid entries
   - Returns cleaned page information with excerpts
   - Supports space filtering and result limiting

2. **`confluence_get_page(page_id)`**
   - Fetches the full content of a specific Confluence page
   - Returns page content, title, and URL

3. **`confluence_create_page(space_key, title, body, parent_id=None)`**
   - Creates a new Confluence page
   - Automatically converts markdown to HTML
   - Supports both storage format (HTML) and wiki format as fallback
   - Returns page ID and URL

4. **`confluence_update_page(page_id, title=None, body=None)`**
   - Updates an existing Confluence page
   - Preserves page versioning

### Query Understanding and Routing

The agent uses intelligent query understanding to handle different types of Confluence-related queries:

- **Meta-questions**: Questions like "What kind of pages can I see?" are detected and handled by showing a sample of available pages
- **Specific searches**: Queries about specific topics are reformulated into effective search queries before searching
- **Export requests**: Queries like "Create a Confluence report" are routed to the export subflow after analysis completion

### Error Handling

The agent provides helpful error messages for common issues:

- **Permissions errors**: Provides troubleshooting steps when the user lacks permission to create pages
- **Space key issues**: Validates space keys and suggests alternatives
- **Format conversion errors**: Automatically falls back to alternative formats if markdown conversion fails

### Workflow Integration

The Confluence integration is seamlessly integrated into the agent workflow:

1. **Classification**: The agent classifies queries to detect Confluence-related intents (`FROM_ANALYSIS` or `FROM_CONFLUENCE`)
2. **Routing**: Based on classification, queries are routed to appropriate Confluence subflows
3. **State Management**: Confluence page IDs and URLs are stored in agent state for future reference
4. **Non-blocking**: If Confluence tools are unavailable, the agent continues to work normally without Confluence features

---

## Future Extensions

- Connecting to additional data sources (Redshift, BigQuery, etc.) using MCP tools
- Enhanced multi-dataset discovery and retrieval
- Multi-agent planning (planner / code writer / verifier)
- Schema retrieval via RAG
- Query decomposition for complex analytics
- Notebook-style report generation
- Additional statistical and machine learning libraries
- Persistent vector database (pgvector) for larger document collections
- Additional document formats (Word, Markdown, etc.)
- Update/append to existing Confluence pages (stretch goal)

This prototype is intentionally simple but complete, demonstrating the viability of LLM-driven data analysis workflows with integrated knowledge management and Confluence documentation.

---

## Setup and Running

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Set Environment Variables

Create a `.env` file with:
```bash
OPENAI_API_KEY=your_openai_api_key
DATA_ANALYSIS_MCP_SERVER_URL=http://localhost:8082/mcp  # Default MCP server URL (must include /mcp path)
LANGGRAPH_SERVER_URL=http://localhost:2024  # Default LangGraph Server URL (used by Streamlit)
LANGGRAPH_ASSISTANT_ID=<your-assistant-uuid>  # Assistant ID (UUID) for LangGraph Server - find it in LangGraph Studio UI

# Optional: For S3-based datasets (e.g., covid_full_grouped)
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET=your_bucket_name

# Optional: For Confluence integration
CONFLUENCE_MCP_SERVER_URL=http://localhost:8083/mcp  # Confluence MCP server URL (if using Confluence integration)
CONFLUENCE_SPACE_KEY_ANALYTICS=ANALYTICS  # Default Confluence space key for analytics reports
```

**Important**: When you run `langgraph dev`, it will automatically create an assistant from your graph. The `assistant_id` is typically a UUID (not the graph name). You can find it by:
1. Opening the LangGraph Studio UI (URL shown when you run `langgraph dev`)
2. Looking at the assistant details in the Studio UI
3. Or checking the server logs when `langgraph dev` starts

Once you have the UUID, set it as `LANGGRAPH_ASSISTANT_ID` in your `.env` file.

### 3. Start the Servers

The system consists of three separate servers that need to be running:

**Terminal 1 - MCP Server:**
```bash
python -m src.mcp_server
```
The MCP server will start on port 8082 by default (configurable via `PORT` environment variable).

**Terminal 2 - LangGraph Server:**
```bash
langgraph dev --config src/langgraph_server/langgraph.json
```
The LangGraph Server will start on port 2024 by default (configurable via `--port` flag).

Alternatively, you can run from the `langgraph_server` directory:
```bash
cd src/langgraph_server
langgraph dev
```
This provides the official LangGraph Server API with endpoints like `/threads`, `/runs`, `/messages`.

**Terminal 3 - Streamlit UI:**
```bash
streamlit run src/app/ui.py
```
The Streamlit app will open in your browser automatically (usually at `http://localhost:8501`).

### 4. Alternative: CLI Mode

You can also run the agent in CLI mode (without the LangGraph Server):

**Interactive mode:**
```bash
python -m src.langgraph_server
```

**Single query:**
```bash
python -m src.langgraph_server "How does the number of patients vary from January to July 2025 in Tokyo?"
```

### 5. Generate Workflow Diagram (Optional)

To generate a visual diagram of the agent workflow:

```bash
python -m src.langgraph_server.generate_diagram
```

Options:
- `--output` or `-o`: Specify output file path (default: `agent_workflow.png`)
- `--format` or `-f`: Specify format - `png`, `svg`, or `mermaid` (default: `png`)

Examples:
```bash
# Generate PNG diagram
python -m src.langgraph_server.generate_diagram

# Generate SVG diagram
python -m src.langgraph_server.generate_diagram --format svg

# Custom output path
python -m src.langgraph_server.generate_diagram --output my_workflow.png
```

---

## Architecture Notes

The project follows a clean separation of concerns with three independent servers:

1. **MCP Server** (`src/mcp_server/`): 
   - Defines and exposes tools via FastMCP
   - Runs on port 8082 (default)
   - Tools are independent and can be reused by other agents or services
   - Supports multiple storage backends (local files, S3) via DatasetStore abstraction

2. **LangGraph Server** (`src/langgraph_server/`): 
   - Defines the agent workflow and consumes MCP tools via HTTP
   - Served via `langgraph dev` command (official LangGraph Server runtime)
   - Runs on port 2024 (default) with endpoints: `/threads`, `/runs`, `/messages`
   - Provides built-in persistence, event streaming, and checkpointing
   - Can also run in CLI mode: `python -m src.langgraph_server`
   - The agent doesn't need to know about tool implementation details, only their interfaces

3. **Streamlit UI** (`src/app/ui.py`):
   - Web-based user interface
   - Connects to LangGraph server via HTTP API
   - Runs on port 8501 (default, Streamlit default)
   - Completely decoupled from the agent implementation

This architecture provides:
- **Separation of concerns**: Each server has a single responsibility
- **Scalability**: Each component can be scaled independently
- **Testability**: Components can be tested in isolation
- **Flexibility**: Easy to swap UI implementations or add new clients
- **Reusability**: Tools and agent can be used by other services
