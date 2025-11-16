# Data Analysis Agent Prototype

This project is a 20-hour prototype demonstrating an LLM-powered data analysis agent that can answer analytical questions by generating Python code, executing it, analyzing the results, and responding in natural language.

The goal is to show how an agent can perform end-to-end analysis on internal datasets (such as S3 or Redshift) in the future. For the prototype, the agent operates on a single local CSV file containing COVID-19 patient counts for each prefecture in Japan.

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
  - Execution Tool (`run_covid_analysis`)  
    - loads the COVID dataset into a predefined dataframe `df`  
    - executes generated Python code safely  
    - returns previews and plots  
  - Dataset Tool (`get_dataset_schema`)  
    - returns column names  
    - datatypes  
    - sample rows  
    - used to condition the agent's code generation  

The architecture follows a separation of concerns pattern where:
- MCP tools are defined in `mcp_server/` and exposed via FastMCP
- LangGraph agent flow is defined in `langgraph_server/` and consumes MCP tools

---

## Repository Structure
```
project-root/
│
├── data/
│ └── newly_confirmed_cases_daily.csv
│
├── src/
│ ├── langgraph_server/
│ │ ├── graph.py # LangGraph agent definition
│ │ ├── prompts.py # Agent prompts and reasoning steps
│ │ ├── mcp_tool_loader.py # Loads tools from MCP server
│ │ └── settings.py # LangGraph server settings
│ │
│ ├── mcp_server/
│ │ ├── server.py # FastMCP server entry point
│ │ ├── analysis_tools.py # MCP tools (get_dataset_schema, run_covid_analysis)
│ │ ├── schema.py # Pydantic schemas for tool inputs/outputs
│ │ └── settings.py # MCP server settings
│ │
│ ├── app/
│ │ └── ui.py # Streamlit UI
│ │
│ ├── main.py # Main entry point
│ └── generate_workflow_diagram.py # Script to generate workflow diagram
│
├── run_mcp_server.py # Script to start the MCP server
└── README.md
```

---

## Tools

### `get_dataset_schema()`

Returns:
- Column names  
- Inferred dtypes  
- First N rows (JSON)

Helps the agent avoid hallucinating nonexistent columns or datatypes.

---

### `run_covid_analysis(code: str)`

Executes generated Python code in a controlled environment.

Execution rules:
- The CSV is loaded as `df`
- Allowed imports: pandas, numpy, matplotlib.pyplot
- Captures:
  - stdout  
  - error  
  - preview of `result_df` if defined  
  - saved plot with timestamped filename (e.g., `plot_20251115_212901.png`) if created  

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
```

### 3. Start the MCP Server

In one terminal, start the MCP server:
```bash
# Option 1: Run as a module (recommended)
python -m src.mcp_server

# Option 2: Use the convenience script
python run_mcp_server.py
```

The MCP server will start on port 8082 by default (configurable via `PORT` environment variable).

### 4. Run the Agent

In another terminal, run the agent:

**Interactive mode:**
```bash
# Option 1: Run as a module (recommended)
python -m src.langgraph_server

# Option 2: Use the convenience wrapper
python -m src.main
```

**Single query:**
```bash
python -m src.langgraph_server "How does the number of patients vary from January to July 2025 in Tokyo?"
```

**Streamlit UI:**

1. Make sure the MCP server is running (see step 3 above)
2. In a new terminal, run:
   ```bash
   streamlit run src/app/ui.py
   ```
   
   The Streamlit app will open in your browser automatically.

---

## Architecture Notes

The project follows a clean separation of concerns:

- **MCP Server** (`src/mcp_server/`): Defines and exposes tools via FastMCP. Tools are independent and can be reused by other agents or services.

- **LangGraph Server** (`src/langgraph_server/`): Defines the agent workflow and consumes MCP tools. The agent doesn't need to know about tool implementation details, only their interfaces.

This architecture makes it easy to:
- Add new tools by extending the MCP server
- Reuse tools across different agents
- Test tools independently
- Scale tools as separate services
