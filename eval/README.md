### Evaluation Harness

This folder contains an automated evaluation harness for the data-analysis-agent with extensible metrics, concurrent execution, and comprehensive visualization.

**Directory Structure:**
```
eval/
├── cases/              # YAML evaluation case files
├── img/                # Generated evaluation plots (eval/img/eval_history.png)
├── schema.py           # Pydantic models for type-safe evaluation
├── metrics.py          # Metric implementations (LLM judge, etc.)
├── prompts.py          # Prompt templates for LLM judge
├── run_eval.py         # Main evaluation runner
├── plot_eval_history.py # Visualization utilities
├── update_history.py    # History tracking utilities
└── langgraph_client.py # HTTP client for LangGraph server
```

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
- `OPENAI_API_KEY` or `CHAT_NODE__api_key` (for evaluation judge LLM)
- `EVAL_JUDGE_MODEL` (optional; defaults to `gpt-4o-mini`)
- `EVAL_JUDGE_PROVIDER` (optional; defaults to `openai`)

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
- Send each `query` to the LangGraph server over HTTP (concurrently by default)
- Apply the configured checks and metrics to the response
- Evaluate each metric criterion separately using LLM-as-judge
- Compute per-case scores and an overall score
- Write a JSON report to `eval_report.json` with detailed metric results
- Exit with non-zero status if the overall score is below `--fail-under`

**Concurrent Execution:**
- By default, cases run concurrently with `min(32, num_cases + 4)` workers
- Use `--max-workers N` to control concurrency
- Each case runs independently, significantly speeding up evaluation

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
- **Checklist-based**: Each criterion is evaluated separately by the LLM judge
- **Satisfaction scores**: Each criterion gets a score from 0-100
- **Weighted average**: Overall score is calculated using criterion weights
- **Standardized scale**: All scores range from 0 to 100
- **Detailed feedback**: Each criterion includes reasoning, strengths, and weaknesses
- **Pydantic validation**: All configurations are validated using Pydantic models

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
python -m eval.plot_eval_history --history eval_history.json --out eval/img/eval_history.png
```

This generates a comprehensive figure (`eval/img/eval_history.png`) with subplots showing:
- **Overall score** over time (with pass/fail indicators)
- **Each metric's trend** over time (0-100 scale) - one subplot per metric
- **Individual case scores** over time (all cases in one plot)

**Plot Features:**
- **Shared x-axis**: All subplots use the same time range and tick marks for easy comparison
- **Consistent formatting**: All labels, fonts, and scales are standardized
- **Comprehensive view**: See all metrics and cases in a single figure
- **Easy trend detection**: Quickly identify improvements or regressions across all metrics

All plots are combined into a single comprehensive figure, making it easy to see improvements across all metrics and cases at a glance. The shared x-axis ensures you can easily compare metrics across the same time periods, even if metrics were added or removed at different points.

#### Architecture

The evaluation system follows clean architecture principles:

- **`eval/schema.py`**: Pydantic models for type-safe configuration and results
  - `Criterion`: Individual evaluation criterion with name, description, and weight
  - `CriterionResult`: Result for each criterion with satisfaction score (0-100)
  - `LLMJudgeConfig`: Validated configuration for LLM judge metric
  - `MetricResult`: Complete metric result with overall score and criterion details

- **`eval/metrics.py`**: Evaluation logic and metric implementations
  - `BaseMetric`: Abstract base class for all metrics
  - `LLMJudgeMetric`: LLM-as-judge implementation with checklist-based evaluation

- **`eval/prompts.py`**: Prompt templates for LLM judge evaluation
  - Uses LangChain's `ChatPromptTemplate` for maintainable prompts

- **`eval/plot_eval_history.py`**: Visualization utilities
  - Generates comprehensive figures with shared x-axis across all subplots

#### Adding Custom Metrics (Advanced)

The metrics system is extensible. You can register custom metrics by extending `BaseMetric`:

```python
from eval.metrics import BaseMetric, register_metric
from eval.schema import MetricResult, CriterionResult
from eval.langgraph_client import LangGraphResult
from typing import Optional

class MyCustomMetric(BaseMetric):
    def evaluate(
        self, 
        query: str, 
        result: LangGraphResult, 
        reference: Optional[str] = None
    ) -> MetricResult:
        # Your evaluation logic
        overall_score = calculate_score(...)  # 0-100 scale
        criterion_results = [...]  # List of CriterionResult objects
        
        return MetricResult(
            metric_name=self.name,
            metric_type="my_custom_metric",
            overall_score=overall_score,
            passed=overall_score >= self.config.get("threshold", 70.0),
            criterion_results=criterion_results,
            details={...},
        )

register_metric("my_custom_metric", MyCustomMetric)
```

Then use it in your YAML cases:

```yaml
metrics:
  - type: my_custom_metric
    name: "custom_score"
    threshold: 70.0  # 0-100 scale
```

#### Report Structure

The evaluation report (`eval_report.json`) includes:

```json
{
  "metadata": {
    "timestamp": "...",
    "git_sha": "...",
    "fail_under": 0.85,
    "cases_dir": "eval/cases"
  },
  "overall_score": 0.85,
  "pass": true,
  "case_results": [
    {
      "id": "case-id",
      "score": 0.9,
      "passed": true,
      "checks": [...],
      "metrics": [
        {
          "metric_name": "quality_score",
          "metric_type": "llm_judge",
          "overall_score": 85.0,
          "passed": true,
          "criterion_results": [
            {
              "criterion_name": "clarity",
              "satisfaction_score": 90.0,
              "reasoning": "...",
              "passed": true
            }
          ]
        }
      ]
    }
  ]
}
```

