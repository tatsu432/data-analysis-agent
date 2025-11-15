# Setup Instructions

## Prerequisites

- Python 3.9 or higher
- OpenAI API key

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Command Line Interface

**Interactive mode:**
```bash
python main.py
```

**Single query mode:**
```bash
python main.py "How does the number of patients vary from January to July 2025 in Tokyo?"
```

### Streamlit UI

```bash
streamlit run app/ui.py
```

### Example Script

```bash
python example_usage.py
```

## Project Structure

```
data-analysis-agent/
├── agent/
│   ├── graph.py          # LangGraph agent definition
│   ├── prompts.py        # Agent prompts
│   └── tools/
│       ├── schema.py     # Dataset schema tool
│       └── execution.py  # Code execution tool
├── app/
│   └── ui.py            # Streamlit UI
├── data/
│   └── newly_confirmed_cases_daily.csv
├── main.py              # CLI entry point
├── example_usage.py     # Example queries
└── requirements.txt     # Dependencies
```

## How It Works

1. User asks a natural language question
2. Agent uses `get_dataset_schema` to understand the data structure
3. Agent generates Python code to answer the question
4. Agent executes code using `run_covid_analysis` tool
5. Agent analyzes results and provides a natural language summary

The agent can handle errors and retry with corrected code automatically.

