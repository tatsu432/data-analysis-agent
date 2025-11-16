# Data Analysis Agent Prototype

This project is a 20-hour prototype demonstrating an LLM-powered data analysis agent that can answer analytical questions by generating Python code, executing it, analyzing the results, and responding in natural language.

The goal is to show how an agent can perform end-to-end analysis on internal datasets (such as S3 or Redshift) in the future. For the prototype, the agent operates on multiple local CSV files including COVID-19 patient counts for Japanese prefectures and patient data.

---

## What This Prototype Demonstrates

The agent can:

1. Interpret natural-language analytical questions  
   Example: “How does the number of patients vary from January to July 2025 in Tokyo?”

2. Plan the required analytical steps (filtering, grouping, aggregating, plotting)

3. Generate executable Python code using pandas and matplotlib

4. Execute the generated code through a controlled tool interface

5. Capture results  
   - dataframe preview  
   - stdout  
   - generated plot (PNG)

6. Understand and summarize the results in natural language

This provides a clear vertical slice of an “LLM analyst” workflow and is designed to be extensible to enterprise data.

---

## Architecture Overview

High-Level Components:

- **LangGraph Server** (`src/langgraph_server/`)  
  - LangGraph agent that interprets queries  
  - writes Python code  
  - retries on errors  
  - summarizes results  
  - loads tools from MCP server via `langchain-mcp-adapters`

- **MCP Server** (`src/mcp_server/`)  
  - FastMCP server exposing data analysis tools
  - Dataset Registry: Centralized registry of available datasets with metadata
  - Execution Tool (`run_analysis`)  
    - loads one or more datasets into dataframes  
    - executes generated Python code safely  
    - returns previews and plots  
    - supports multi-dataset analysis
  - Dataset Tools:
    - `list_datasets`: Lists all available datasets with their IDs and descriptions
    - `get_dataset_schema(dataset_id)`: Returns column names, datatypes, and sample rows for a specific dataset
  - Legacy Tool (`run_covid_analysis`): Deprecated wrapper for backwards compatibility  

The architecture follows a separation of concerns pattern where:
- MCP tools are defined in `mcp_server/` and exposed via FastMCP
- LangGraph agent flow is defined in `langgraph_server/` and consumes MCP tools

---

## Repository Structure
```
project-root/
│
├── data/
│ ├── newly_confirmed_cases_daily.csv
│ ├── jpm_patient_data.csv
│ ├── jamdas_patient_data.csv
│ └── mr_activity_data.csv
│
├── src/
│ ├── langgraph_server/
│ │ ├── graph.py # LangGraph agent definition
│ │ ├── prompts.py # Agent prompts and reasoning steps
│ │ ├── mcp_tool_loader.py # Loads tools from MCP server
│ │ ├── settings.py # LangGraph server settings
│ │ ├── __main__.py # Entry point for CLI mode
│ │ └── generate_diagram.py # Script to generate workflow diagram
│ │
│ ├── mcp_server/
│ │ ├── server.py # FastMCP server entry point
│ │ ├── analysis_tools.py # MCP tools (list_datasets, get_dataset_schema, run_analysis)
│ │ ├── datasets_registry.py # Dataset registry with metadata
│ │ ├── schema.py # Pydantic schemas for tool inputs/outputs
│ │ ├── settings.py # MCP server settings
│ │ └── __main__.py # Entry point for running the MCP server
│ │
│ ├── app/
│ │ └── ui.py # Streamlit UI
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
- File paths

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
- Allowed imports: pandas, numpy, matplotlib.pyplot
- Date columns are automatically converted to datetime
- Captures:
  - stdout  
  - error  
  - preview of `result_df` if defined  
  - saved plot with timestamped filename (e.g., `plot_20251115_212901.png`) if created

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

### `run_covid_analysis(code: str)` (DEPRECATED)

**DEPRECATED**: This tool is deprecated and will be removed in a future version. Use `run_analysis` instead.

Backwards-compatible wrapper that calls `run_analysis` with the COVID dataset. Kept for compatibility with existing code.  

---

## Agent Behavior

The agent follows this loop:

1. Interpret the natural language query
2. Plan the analytical steps in plain text
3. Generate Python code
4. Call the execution tool
5. Inspect returned results
6. Summarize findings for the user

If execution errors occur:
- The agent analyzes the traceback
- Regenerates corrected code
- Retries once

---

## Example Queries

You can ask:

- How does the number of patients vary from January to July 2022 in Tokyo?
- Generate and compare the line plots of the number of patients from January to August 2022 in Tokyo, Chiba, Saitama, Kanagawa.
- What characteristics does the patient count data have overall?
- Can you model the Tokyo's covid case and tell me the model clearly?
- Can you compare the each product's number of patients over the time for GP only?
- Can you generate the line plots of the number of the patients for each product only for those at risk over the time?
- Can you create a regression model where we predict the number of patient for LAGEVRIO by the MR activities? Tell me the fitted model and MAPE.
- Generate the line plots of the number of those who are recovered from COVID over the time for South Korea, Canada, France, and US.
---



## Future Extensions

- Connecting to S3 or Redshift using MCP tools
- Multi-dataset discovery and retrieval
- Multi-agent planning (planner / code writer / verifier)
- Schema retrieval via RAG
- Query decomposition for complex analytics
- Notebook-style report generation

This prototype is intentionally simple but complete, demonstrating the viability of LLM-driven data analysis workflows.

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
