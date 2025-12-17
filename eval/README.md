### Evaluation Harness

This folder contains an automated evaluation harness for the data-analysis-agent.

The evaluation flow is:

- **Start the Unified MCP server** in one terminal:

```bash
python -m src.mcp_server
```

- **Start the LangGraph server** in a second terminal (HTTP mode):

```bash
langgraph dev --config src/langgraph_server/langgraph.json
```

Make sure the following environment variables are set (typically via `.env`):

- `MCP_SERVER_URL` (e.g. `http://localhost:8082/mcp`)
- `LANGGRAPH_SERVER_URL` (e.g. `http://localhost:2024`)
- `LANGGRAPH_ASSISTANT_ID` (optional; if omitted, the client will look up/create one)
- LLM API key(s) as required by your `CHAT_NODE__...` configuration

Then **run the evaluation runner** from the project root:

```bash
python -m eval.run_eval \
  --cases eval/cases \
  --out eval_report.json \
  --fail-under 0.85
```

The runner will:

- Load YAML cases from `eval/cases/`
- Send each `query` to the LangGraph server over HTTP
- Apply the configured checks to the response
- Compute per-case scores and an overall score
- Write a JSON report to `eval_report.json`
- Exit with non-zero status if the overall score is below `--fail-under`

