# Root Cause Analysis: Code Generation Node Loop Issue

## Problem Description
When using open source models (Qwen, GPT-OSS), the code generation node repeatedly calls `get_dataset_schema` instead of calling `run_analysis`, creating an infinite loop.

## Root Causes Identified

### 1. **Routing Logic Creates Loop** (Primary Issue)
**Location**: `graph.py` lines 2489-2524 (`route_from_tools` function)

**Problem**: 
- When `get_dataset_schema` is called, the routing logic checks if `run_analysis` has been called
- If not, it routes back to `code_generation` node
- This creates a loop: `code_generation` → `get_dataset_schema` → `tools` → `code_generation` → `get_dataset_schema` → ...

**Code Issue**:
```python
if last_tool_message.name == "get_dataset_schema":
    if not has_run_analysis:
        return "code_generation"  # Routes back, creating loop
```

**Why it loops**: The routing logic doesn't check if:
- The same dataset's schema was already retrieved in the current cycle
- The model is stuck calling `get_dataset_schema` repeatedly
- There's a maximum retry limit for schema calls

### 2. **Schema Tracking Not Robust Enough**
**Location**: `graph.py` lines 624-651 (schema tracking in `code_generation_node`)

**Problem**:
- The code tracks which datasets have schema by looking at tool messages
- However, it doesn't prevent the model from calling `get_dataset_schema` again for the same dataset
- The tracking might not work correctly if the model doesn't see schema information in context

**Code Issue**:
```python
schemas_found = {}  # Tracks schemas, but doesn't prevent re-calling
# No check to prevent calling get_dataset_schema for datasets that already have schema
```

### 3. **Prompt Doesn't Prevent Repeated Schema Calls**
**Location**: `prompts.py` lines 397-520 (`CODE_GENERATION_PROMPT`)

**Problem**:
- The prompt says "If you don't have schema information, call `get_dataset_schema` first"
- But it doesn't explicitly:
  - Prevent calling `get_dataset_schema` multiple times for the same dataset
  - Check if schema information is already in the conversation history
  - Force `run_analysis` to be called immediately after getting schema

**Missing Instructions**:
- "DO NOT call `get_dataset_schema` if you already have schema information for a dataset"
- "After calling `get_dataset_schema`, you MUST immediately call `run_analysis` - do NOT call `get_dataset_schema` again"
- "If you've already called `get_dataset_schema` for a dataset, use that schema information to generate code"

### 4. **Open Source Models Prefer "Safe" Tool Calls**
**Behavior**: With `tool_choice="required"`, open source models are forced to make a tool call. They tend to:
- Prefer `get_dataset_schema` because it's "safer" (just information gathering, no code generation)
- Avoid `run_analysis` because it requires generating code (more complex, higher risk)
- Not properly understand the workflow: get schema → generate code → call run_analysis

**Why**: Open source models may not have strong instruction-following capabilities compared to GPT-4/Claude, so they default to the "easiest" tool call.

### 5. **No Loop Detection Mechanism**
**Problem**: There's no mechanism to detect when the model is stuck in a loop:
- No counter for how many times `get_dataset_schema` has been called
- No check to see if the same dataset's schema was retrieved multiple times
- No maximum retry limit for schema calls before forcing `run_analysis`

## Proposed Fixes

### Fix 1: Add Loop Detection and Prevention
- Track how many times `get_dataset_schema` has been called in the current code generation cycle
- If `get_dataset_schema` is called more than once for the same dataset, force `run_analysis`
- Add a maximum retry limit (e.g., 2 schema calls max before forcing `run_analysis`)

### Fix 2: Improve Schema Tracking
- Explicitly track which datasets have schema information in the current cycle
- Pass this information to the model in the system message
- Prevent the model from calling `get_dataset_schema` for datasets that already have schema

### Fix 3: Strengthen Prompt Instructions
- Add explicit instruction: "DO NOT call `get_dataset_schema` if schema information is already available"
- Add: "After calling `get_dataset_schema`, you MUST call `run_analysis` - do NOT call `get_dataset_schema` again"
- Add: "If you've already retrieved schema for a dataset, use that information to generate code immediately"

### Fix 4: Smarter Routing Logic
- Check if the same dataset's schema was already retrieved in the current cycle
- If yes, route to `code_generation` but with a stronger prompt forcing `run_analysis`
- Add a flag to track if we're in a loop and force `run_analysis` if detected

### Fix 5: Add Explicit Schema Information to Context
- When routing back to `code_generation` after `get_dataset_schema`, explicitly include the schema information in the system message
- This ensures the model sees the schema even if it's not properly parsing the conversation history

## Implementation Priority

1. **High Priority**: Fix 1 (Loop Detection) + Fix 4 (Smarter Routing) - Prevents infinite loops
2. **Medium Priority**: Fix 2 (Schema Tracking) + Fix 5 (Explicit Schema) - Helps models make better decisions
3. **Low Priority**: Fix 3 (Prompt Strengthening) - Helps but may not be sufficient alone

## Testing Strategy

1. Test with open source models (Qwen, GPT-OSS) that previously caused loops
2. Verify that `get_dataset_schema` is not called more than once per dataset
3. Verify that `run_analysis` is called after schema retrieval
4. Test edge cases: multiple datasets, schema already available, etc.

