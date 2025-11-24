# Fixes Implemented for Code Generation Loop Issue

## Summary
Fixed the infinite loop issue where open source models repeatedly called `get_dataset_schema` instead of `run_analysis`.

## Changes Made

### 1. Loop Detection in Code Generation Node
**File**: `src/langgraph_server/graph.py`
**Location**: `code_generation_node` function (lines ~624-760)

**Changes**:
- Added `schema_call_count` dictionary to track how many times `get_dataset_schema` was called per dataset
- Added loop detection logic that warns when a dataset's schema is called more than once
- Added explicit system message warnings when loops are detected, forcing the model to call `run_analysis`

**Key Code**:
```python
schema_call_count = {}  # Track how many times get_dataset_schema was called per dataset
# ... tracking logic ...
if datasets_with_repeated_schema_calls:
    logger.warning(f"LOOP DETECTED: get_dataset_schema called multiple times...")
    system_parts.append("🚨 CRITICAL - LOOP DETECTION: ... DO NOT call get_dataset_schema again...")
```

### 2. Improved Routing Logic with Loop Prevention
**File**: `src/langgraph_server/graph.py`
**Location**: `route_from_tools` function (lines ~2489-2550)

**Changes**:
- Added loop detection in routing logic that counts how many times `get_dataset_schema` was called for the same dataset
- If a dataset's schema was called more than once, logs a warning and routes to `code_generation` with a flag to force `run_analysis`
- Improved context detection to better identify code generation context

**Key Code**:
```python
# Count how many times this dataset's schema was called
schema_call_count = 0
# ... counting logic ...
if schema_call_count > 1:
    logger.warning(f"LOOP DETECTED: get_dataset_schema called {schema_call_count} times...")
    return "code_generation"  # Route with loop detection flag
```

### 3. Strengthened Prompt Instructions
**File**: `src/langgraph_server/prompts.py`
**Location**: `CODE_GENERATION_PROMPT` (lines ~397-520)

**Changes**:
- Added explicit loop prevention rules (rules 7-8)
- Added warning emojis (🚨) to highlight critical instructions
- Added instruction to check conversation history before calling `get_dataset_schema`
- Clarified workflow to prevent going back to step 1 after getting schema

**Key Additions**:
- Rule 7: "DO NOT call get_dataset_schema multiple times for the same dataset"
- Rule 8: "After calling get_dataset_schema, you MUST call run_analysis next"
- Workflow step 4: "DO NOT go back to step 1 after step 2"

### 4. Enhanced Schema Tracking
**File**: `src/langgraph_server/graph.py`
**Location**: `code_generation_node` function

**Changes**:
- Improved schema tracking to count calls per dataset
- Added explicit warnings in system messages when schema is already available
- Added instruction to use existing schema information instead of calling `get_dataset_schema` again

## How It Works

1. **First Schema Call**: When `get_dataset_schema` is called for a dataset, it's tracked in `schema_call_count`
2. **Loop Detection**: If the same dataset's schema is called again, the count exceeds 1, triggering loop detection
3. **Warning Injection**: When a loop is detected, a strong warning is injected into the system message forcing `run_analysis`
4. **Routing**: The routing logic detects loops and routes back to `code_generation` with loop detection context
5. **Prompt Enforcement**: The strengthened prompt explicitly prevents repeated schema calls

## Testing Recommendations

1. **Test with Open Source Models**: Test with Qwen, GPT-OSS models that previously caused loops
2. **Verify Loop Prevention**: Ensure `get_dataset_schema` is not called more than once per dataset
3. **Verify run_analysis is Called**: Ensure `run_analysis` is called after schema retrieval
4. **Test Edge Cases**:
   - Multiple datasets (some with schema, some without)
   - Schema already available from main agent
   - Model attempts to call `get_dataset_schema` multiple times

## Expected Behavior After Fix

**Before Fix**:
```
code_generation → get_dataset_schema → tools → code_generation → get_dataset_schema → tools → ... (infinite loop)
```

**After Fix**:
```
code_generation → get_dataset_schema → tools → code_generation (with loop detection) → run_analysis → tools → agent
```

## Additional Notes

- The loop detection is per-dataset, so calling `get_dataset_schema` for different datasets is still allowed
- The warnings use emojis (🚨, ⚠️) to make them more visible to the model
- The fixes are backward compatible and don't affect models that were already working correctly

