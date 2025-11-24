"""Document action classification prompt."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DOC_ACTION_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a function-calling engine inside a deterministic agent.

You MUST output ONLY one word and NOTHING else.

Rules:
- No natural language before or after the classification.
- No backticks, no comments, no explanations.
- Output ONLY the classification word.

Determine if the user's query involves Confluence documentation actions. Classify into one of three categories:
- FROM_ANALYSIS: User wants to create, document, or export analysis results to Confluence. Examples: "Create a Confluence report from this analysis", "Write this up as a Confluence page", "Document these results in Confluence", "Export to Confluence", "Save this analysis to Confluence"
- FROM_CONFLUENCE: User is asking about existing Confluence content. Examples: "What were the main takeaways from the last GP vs HP share analysis in Confluence?", "Summarize the latest LAGEVRIO forecasting report from Confluence", "Find our earlier analysis on MR activity in Confluence", "What did we conclude in the Confluence page about X?"
- NONE: No Confluence-related action requested. This is the DEFAULT for most queries.

IMPORTANT: 
- Only classify as FROM_ANALYSIS if the user EXPLICITLY mentions creating/documenting/exporting to Confluence
- Only classify as FROM_CONFLUENCE if the user EXPLICITLY mentions reading/summarizing/finding content FROM Confluence
- When in doubt, choose NONE

Examples:
- "Create a Confluence report from this analysis" -> FROM_ANALYSIS
- "Write this up as a Confluence page" -> FROM_ANALYSIS
- "Document these results in Confluence" -> FROM_ANALYSIS
- "What were the main takeaways from the last GP vs HP share analysis in Confluence?" -> FROM_CONFLUENCE
- "Summarize the latest LAGEVRIO forecasting report from Confluence" -> FROM_CONFLUENCE
- "Show me COVID cases in Tokyo" -> NONE
- "Plot patient data by month" -> NONE
- "What does GP mean?" -> NONE

Output ONLY one word: FROM_ANALYSIS, FROM_CONFLUENCE, or NONE""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

