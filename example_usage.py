"""Example usage of the data analysis agent."""
import asyncio
from main import run_analysis


async def main():
    """Run example queries."""
    examples = [
        "What characteristics does the patient count data have overall?",
        "How does the number of patients vary from January to July 2025 in Tokyo?",
        "Show me the top 5 prefectures with the highest total cases in 2024.",
    ]
    
    for query in examples:
        await run_analysis(query, thread_id="example-session")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

