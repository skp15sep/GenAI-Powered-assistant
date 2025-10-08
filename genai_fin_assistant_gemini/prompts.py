from typing import List, Dict

SUMMARY_PROMPT_SYSTEM = (
    "You are a helpful financial assistant. "
    "You analyze transaction summaries and provide concise, personalized financial insights, "
    "spending breakdowns, and saving suggestions."
)

def build_insight_prompt(transactions_summary: Dict, user_query: str) -> List[Dict]:
    """
    Build structured prompt messages for Gemini.
    """
    summary_text = "Transaction Summary:\n"
    for k, v in transactions_summary.items():
        summary_text += f"- {k}: {v}\n"

    user_message = f"""
User Query: {user_query}

{summary_text}

Please answer in this format:
1️⃣ Short Answer (1–2 lines)
2️⃣ 3 Actionable Insights
3️⃣ 2 Spending Trends
Make it concise, data-backed, and human-friendly.
"""
    return [
        {"role": "system", "content": SUMMARY_PROMPT_SYSTEM},
        {"role": "user", "content": user_message},
    ]