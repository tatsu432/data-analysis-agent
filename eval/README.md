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

**Performance Options:**

- `--max-workers N`: Run evaluations concurrently with N workers (default: auto-detected based on number of cases)
- `--max-cases N`: Limit the number of cases to run (useful for testing)

Example with concurrent execution:

```bash
python -m eval.run_eval \
  --cases eval/cases \
  --out eval_report.json \
  --fail-under 0.85 \
  --max-workers 8
```

The runner will:

- Load YAML cases from `eval/cases/`
- Send each `query` to the LangGraph server over HTTP
- Apply the configured checks and metrics to the response
- Compute per-case scores and an overall score
- Write a JSON report to `eval_report.json`
- Exit with non-zero status if the overall score is below `--fail-under`

### Evaluation Metrics System

The evaluation system supports both **legacy checks** (simple rule-based) and **LLM-as-judge metrics** for sophisticated quality evaluation.

#### Legacy Checks

Simple rule-based checks are still supported:

```yaml
checks:
  - type: contains
    text: "data"
  - type: regex
    pattern: "\\d+"
  - type: json_path_numeric_range
    path: "score"
    min: 0.5
    max: 1.0
  - type: artifact_exists
    glob: "img/plot_*.png"
```

#### LLM-as-Judge Metrics (Checklist-based)

The LLM-as-judge metric uses an LLM to evaluate response quality using a checklist of criteria. Each criterion is evaluated separately and scored from 0 to 100. The overall score is calculated as a weighted average of criterion satisfaction scores.

```yaml
metrics:
  - type: llm_judge
    name: "quality_score"
    criteria:
      - name: "clarity"
        description: "Is the response clear and easy to understand?"
        weight: 1.0
      - name: "accuracy"
        description: "Is the information accurate and correct?"
        weight: 1.5
      - name: "completeness"
        description: "Does the response cover all key aspects?"
        weight: 1.2
      - name: "relevance"
        description: "Is the response relevant to the query?"
        weight: 1.0
    threshold: 70.0  # Overall score from 0-100 (not 0-1)
    model: "gpt-4o-mini"  # Optional: defaults to EVAL_JUDGE_MODEL env var
    provider: "openai"    # Optional: defaults to EVAL_JUDGE_PROVIDER env var
```

**Key Features:**
- **Checklist-based**: Each criterion is evaluated separately
- **Satisfaction scores**: Each criterion gets a score from 0-100
- **Weighted average**: Overall score is calculated using criterion weights
- **Standardized scale**: All scores range from 0 to 100

You can also provide an optional reference answer for comparison:

```yaml
reference: "Expected answer text here"
metrics:
  - type: llm_judge
    name: "quality_score"
    criteria:
      - name: "match_reference"
        description: "How well does the response match the reference answer?"
        weight: 2.0
    threshold: 80.0
```

#### Score Calculation

- If both checks and metrics are present: **30% checks, 70% metrics**
- If only checks: uses checks only
- If only metrics: uses metrics only

#### Tracking Metrics Over Time

After running evaluations, update the history:

```bash
python -m eval.update_history --report eval_report.json --history eval_history.json
```

Then visualize trends:

```bash
python -m eval.plot_eval_history --history eval_history.json --out img/eval_history.png
```

This generates:
- `eval_history.png` - Overall score over time
- `eval_history_metrics.png` - Individual metric trends
- `eval_history_cases.png` - Individual case scores

#### Adding Custom Metrics (Advanced)

The metrics system is extensible. You can register custom metrics by extending `BaseMetric`:

```python
from eval.metrics import BaseMetric, register_metric, MetricResult
from eval.langgraph_client import LangGraphResult

class MyCustomMetric(BaseMetric):
    def evaluate(self, query: str, result: LangGraphResult, reference: Optional[str] = None) -> MetricResult:
        # Your evaluation logic
        score = calculate_score(...)
        return self._create_result(score=score, details={...})

register_metric("my_custom_metric", MyCustomMetric)
```

Then use it in your YAML cases:

```yaml
metrics:
  - type: my_custom_metric
    name: "custom_score"
    threshold: 0.7
```

