# processor.py
import pandas as pd
from typing import Dict, Any


def load_transactions_csv(path: str) -> pd.DataFrame:
    """Load transaction data from CSV and ensure 'date' is datetime."""
    df = pd.read_csv(path)

    # Convert 'date' column to datetime (force errors to NaT)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    # Drop rows where 'date' could not be parsed
    df = df.dropna(subset=["date"])

    return df


def summarize_transactions(df: pd.DataFrame, currency: str = "INR") -> Dict[str, Any]:
    """Compute spending summary and top categories."""
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    now = pd.Timestamp.now()
    last_30 = df[df["date"] >= (now - pd.Timedelta(days=30))]

    summary = {
        "currency": currency,
        "total_spent_30d": f"{last_30['amount'].sum():.2f} {currency}",
        "total_spent_all": f"{df['amount'].sum():.2f} {currency}",
        "transaction_count_30d": int(last_30.shape[0]),
        "avg_daily_spend_30d": f"{(last_30['amount'].sum()/30):.2f} {currency}",
    }

    top_categories = last_30.groupby(
        "category")["amount"].sum().sort_values(ascending=False).head(3)
    summary["top_categories_30d"] = {
        k: f"{v:.2f} {currency}" for k, v in top_categories.items()}
    return summary


def answer_user_query(openai_client, df: pd.DataFrame, user_query: str) -> str:
    """Generate response using OpenAI model."""
    from prompts import build_insight_prompt
    summary = summarize_transactions(df)
    messages = build_insight_prompt(summary, user_query)
    response = openai_client.chat(messages)
    return response
