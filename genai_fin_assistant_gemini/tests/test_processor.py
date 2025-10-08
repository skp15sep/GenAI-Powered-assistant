import pandas as pd
from processor import summarize_transactions

def test_summary_basic():
    data = {
        "date": ["2025-09-01", "2025-09-02"],
        "description": ["Grocery", "Coffee"],
        "category": ["groceries", "food"],
        "amount": [100, 50],
        "currency": ["INR", "INR"]
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    summary = summarize_transactions(df)
    assert "total_spent_all" in summary
    assert summary["currency"] == "INR"