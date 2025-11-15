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

- LangGraph agent  
  - interprets queries  
  - writes Python code  
  - retries on errors  
  - summarizes results  

- Execution Tool (`run_covid_analysis`)  
  - loads the COVID dataset into a predefined dataframe `df`  
  - executes generated Python code safely  
  - returns previews and plots  

- Dataset Tool (`get_dataset_schema`)  
  - returns column names  
  - datatypes  
  - sample rows  
  - used to condition the agent’s code generation  

(Optional: these tools can be served via an MCP server later if needed.)

---

## Repository Structure
```
project-root/
│
├── data/
│ └── newly_confirmed_cases_daily.csv
│
├── src/
│ ├── agent/
│ │ ├── graph.py # LangGraph definition
│ │ ├── prompts.py # Agent prompts and reasoning steps
│ │ └── tools/
│ │ │ ├── execution.py # run_covid_analysis tool
│ │ │ └── schema.py # get_dataset_schema tool
│ │
│ ├── app/
│ │ └── ui.py # Optional Streamlit or FastAPI UI
│ │
│ ├── main.py # Main entry point
│ └── generate_workflow_diagram.py # Script to generate workflow diagram
│
└── README.md
```

---

## Tools

### get_dataset_schema()

Returns:
- Column names  
- Inferred dtypes  
- First N rows (JSON)

Helps the agent avoid hallucinating nonexistent columns or datatypes.

---

### run_covid_analysis(code: str)

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

- “How does the number of patients vary from January to July 2022 in Tokyo?”
- “Generate and compare the line plots of the number of patients from January to August 2022 in Tokyo, Chiba, Saitama, Kanagawa.”
- “What characteristics does the patient count data have overall?”
- "Can you model the Tokyo's covid case and tell me the model clearly?"

---



## Future Extensions

- Connecting to S3 or Redshift using MCP tools
- Multi-dataset discovery and retrieval
- Multi-agent planning (planner / code writer / verifier)
- Schema retrieval via RAG
- Query decomposition for complex analytics
- Notebook-style report generation

This prototype is intentionally simple but complete, demonstrating the viability of LLM-driven data analysis workflows.
